//! C-compatible API for gaming engine integration
//!
//! This module provides C FFI bindings for integrating VoiRS spatial audio
//! with game engines including Unity, Unreal Engine, and Godot.

use super::audio::GamingAudioManager;
use super::config::GamingConfig;
use super::types::{AttenuationCurve, AttenuationSettings, AudioCategory, GameAudioSource};
use crate::types::Position3D;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};

// C-compatible API for gaming engines
extern "C" {
    // These would be implemented in a separate C binding file
}

/// C API functions for Unity/Unreal integration
///
/// # Safety
///
/// This function is unsafe because it:
/// - Dereferences a raw pointer (`config_json`)
/// - Assumes the pointer points to a valid null-terminated C string
/// - The caller must ensure the pointer is valid for the duration of the call
/// - The returned pointer must be freed with `voirs_gaming_destroy_manager`
#[no_mangle]
pub unsafe extern "C" fn voirs_gaming_create_manager(config_json: *const c_char) -> *mut c_void {
    if config_json.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let config_str = match CStr::from_ptr(config_json).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        let config: GamingConfig = serde_json::from_str(config_str).unwrap_or_default();

        let runtime = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(_) => return std::ptr::null_mut(),
        };

        match runtime.block_on(GamingAudioManager::new(config)) {
            Ok(manager) => Box::into_raw(Box::new(manager)) as *mut c_void,
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// Destroy a gaming audio manager instance
#[no_mangle]
pub extern "C" fn voirs_gaming_destroy_manager(manager: *mut c_void) {
    if !manager.is_null() {
        unsafe {
            let _ = Box::from_raw(manager as *mut GamingAudioManager);
        }
    }
}

/// Create a new audio source in the gaming manager
#[no_mangle]
pub extern "C" fn voirs_gaming_create_source(
    manager: *mut c_void,
    category: c_int,
    priority: c_int,
    x: c_float,
    y: c_float,
    z: c_float,
) -> c_int {
    if manager.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let category = match category {
            0 => AudioCategory::Player,
            1 => AudioCategory::Npc,
            2 => AudioCategory::Environment,
            3 => AudioCategory::Music,
            4 => AudioCategory::Sfx,
            5 => AudioCategory::Ui,
            6 => AudioCategory::Voice,
            7 => AudioCategory::Ambient,
            _ => AudioCategory::Sfx,
        };

        let position = Position3D::new(x, y, z);

        match manager.create_source(category, priority as u8, position) {
            Ok(source) => source.id as c_int,
            Err(_) => -1,
        }
    }
}

/// Play an audio source in the gaming manager
#[no_mangle]
pub extern "C" fn voirs_gaming_play_source(manager: *mut c_void, source_id: c_int) -> c_int {
    if manager.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let source = GameAudioSource {
            id: source_id as u32,
            category: AudioCategory::Sfx, // Would need to track this
            priority: 128,
        };

        match manager.play_source(source) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// Update listener position and orientation in the gaming manager
#[no_mangle]
pub extern "C" fn voirs_gaming_update_listener(
    manager: *mut c_void,
    x: c_float,
    y: c_float,
    z: c_float,
    forward_x: c_float,
    forward_y: c_float,
    forward_z: c_float,
) -> c_int {
    if manager.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let position = Position3D::new(x, y, z);
        let orientation = (forward_x, forward_y, forward_z);

        match manager.update_listener(position, orientation) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// Unity-specific C API functions
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn voirs_unity_initialize_manager(
    manager: *mut c_void,
    unity_config_json: *const c_char,
) -> c_int {
    if manager.is_null() || unity_config_json.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let config_str = match CStr::from_ptr(unity_config_json).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        };

        // Parse Unity-specific configuration
        // In a real implementation, this would parse Unity-specific settings
        tracing::info!("Initializing Unity manager with config: {}", config_str);

        let runtime = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(_) => return -1,
        };

        match runtime.block_on(manager.initialize_unity()) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// Set Unity audio listener transform
#[no_mangle]
pub extern "C" fn voirs_unity_set_audio_listener_transform(
    manager: *mut c_void,
    position_x: c_float,
    position_y: c_float,
    position_z: c_float,
    rotation_x: c_float,
    rotation_y: c_float,
    rotation_z: c_float,
    rotation_w: c_float,
) -> c_int {
    if manager.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let position = Position3D::new(position_x, position_y, position_z);

        // Convert quaternion to forward vector (simplified)
        let forward_x = 2.0 * (rotation_x * rotation_z + rotation_w * rotation_y);
        let forward_y = 2.0 * (rotation_y * rotation_z - rotation_w * rotation_x);
        let forward_z = 1.0 - 2.0 * (rotation_x * rotation_x + rotation_y * rotation_y);

        let orientation = (forward_x, forward_y, forward_z);

        match manager.update_listener(position, orientation) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// Create Unity audio source
#[no_mangle]
pub extern "C" fn voirs_unity_create_audiosource(
    manager: *mut c_void,
    gameobject_id: c_int,
    clip_id: c_int,
    x: c_float,
    y: c_float,
    z: c_float,
    volume: c_float,
    pitch: c_float,
) -> c_int {
    if manager.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let position = Position3D::new(x, y, z);

        // Map Unity parameters to VoiRS parameters
        let category = AudioCategory::Sfx; // Default category
        let priority = ((1.0 - volume) * 255.0) as u8; // Higher volume = higher priority

        match manager.create_source(category, priority, position) {
            Ok(source) => {
                tracing::info!(
                    "Created Unity AudioSource: GameObject={}, Clip={}, Source={}, Volume={}, Pitch={}",
                    gameobject_id, clip_id, source.id, volume, pitch
                );
                source.id as c_int
            }
            Err(_) => -1,
        }
    }
}

/// Unreal Engine-specific C API functions
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn voirs_unreal_initialize_manager(
    manager: *mut c_void,
    unreal_config_json: *const c_char,
) -> c_int {
    if manager.is_null() || unreal_config_json.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let config_str = match CStr::from_ptr(unreal_config_json).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        };

        // Parse Unreal-specific configuration
        tracing::info!("Initializing Unreal manager with config: {}", config_str);

        let runtime = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(_) => return -1,
        };

        match runtime.block_on(manager.initialize_unreal()) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// Set Unreal Engine audio listener transform
#[no_mangle]
pub extern "C" fn voirs_unreal_set_audio_listener_transform(
    manager: *mut c_void,
    location_x: c_float,
    location_y: c_float,
    location_z: c_float,
    rotation_pitch: c_float,
    rotation_yaw: c_float,
    rotation_roll: c_float,
) -> c_int {
    if manager.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let position = Position3D::new(location_x, location_y, location_z);

        // Convert Unreal rotation (pitch, yaw, roll) to forward vector
        let yaw_rad = rotation_yaw.to_radians();
        let pitch_rad = rotation_pitch.to_radians();

        let forward_x = yaw_rad.cos() * pitch_rad.cos();
        let forward_y = yaw_rad.sin() * pitch_rad.cos();
        let forward_z = pitch_rad.sin();

        let orientation = (forward_x, forward_y, forward_z);

        match manager.update_listener(position, orientation) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// Create Unreal Engine audio component
#[no_mangle]
pub extern "C" fn voirs_unreal_create_audio_component(
    manager: *mut c_void,
    actor_id: c_int,
    sound_wave_id: c_int,
    location_x: c_float,
    location_y: c_float,
    location_z: c_float,
    volume_multiplier: c_float,
    pitch_multiplier: c_float,
    attenuation_distance: c_float,
) -> c_int {
    if manager.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let position = Position3D::new(location_x, location_y, location_z);

        // Map Unreal parameters to VoiRS parameters
        let category = AudioCategory::Sfx; // Default category
        let priority = ((1.0 - volume_multiplier) * 255.0) as u8;

        match manager.create_source(category, priority, position) {
            Ok(source) => {
                tracing::info!(
                    "Created Unreal AudioComponent: Actor={}, SoundWave={}, Source={}, Volume={}, Pitch={}, Attenuation={}",
                    actor_id, sound_wave_id, source.id, volume_multiplier, pitch_multiplier, attenuation_distance
                );
                source.id as c_int
            }
            Err(_) => -1,
        }
    }
}

/// Set Unreal Engine audio component location
#[no_mangle]
pub extern "C" fn voirs_unreal_set_audio_component_location(
    manager: *mut c_void,
    source_id: c_int,
    location_x: c_float,
    location_y: c_float,
    location_z: c_float,
) -> c_int {
    if manager.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let source = GameAudioSource {
            id: source_id as u32,
            category: AudioCategory::Sfx,
            priority: 128,
        };
        let position = Position3D::new(location_x, location_y, location_z);

        match manager.update_source_position(source, position) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// Engine-agnostic utility functions
#[no_mangle]
pub extern "C" fn voirs_gaming_set_source_attenuation(
    manager: *mut c_void,
    source_id: c_int,
    min_distance: c_float,
    max_distance: c_float,
    attenuation_curve: c_int,
    attenuation_factor: c_float,
) -> c_int {
    if manager.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);
        let mut sources = match manager.sources.lock() {
            Ok(s) => s,
            Err(_) => return -1,
        };

        if let Some(source_data) = sources.get_mut(&(source_id as u32)) {
            let curve = match attenuation_curve {
                0 => AttenuationCurve::Linear,
                1 => AttenuationCurve::Logarithmic,
                2 => AttenuationCurve::Exponential,
                3 => AttenuationCurve::Custom,
                _ => AttenuationCurve::Logarithmic,
            };

            source_data.attenuation = AttenuationSettings {
                min_distance,
                max_distance,
                curve,
                factor: attenuation_factor,
            };

            0
        } else {
            -1
        }
    }
}

/// Get performance metrics from gaming manager (C FFI)
#[no_mangle]
pub extern "C" fn voirs_gaming_get_performance_metrics(
    manager: *mut c_void,
    metrics_out: *mut c_void,
) -> c_int {
    if manager.is_null() || metrics_out.is_null() {
        return -1;
    }

    unsafe {
        let manager = &*(manager as *const GamingAudioManager);

        match manager.get_metrics() {
            Ok(metrics) => {
                // In a real implementation, this would copy metrics to the output struct
                tracing::info!(
                    "Performance metrics: {}ms audio time, {} active sources, {:.1} FPS",
                    metrics.audio_time_ms,
                    metrics.active_sources,
                    metrics.fps
                );
                0
            }
            Err(_) => -1,
        }
    }
}
