//! Voice management C API functions.
//!
//! This module provides functions for managing and selecting voices
//! in the VoiRS FFI C API.

use crate::{get_pipeline_manager, get_runtime, set_last_error, VoirsErrorCode};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;

/// Detailed voice information structure for C API (includes gender)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VoirsVoiceInfoDetailed {
    /// Voice ID (null-terminated string)
    pub id: *mut c_char,
    /// Voice name (null-terminated string)
    pub name: *mut c_char,
    /// Language code (null-terminated string)
    pub language: *mut c_char,
    /// Voice gender (0 = unknown, 1 = male, 2 = female, 3 = neutral)
    pub gender: c_int,
    /// Voice quality (0 = low, 1 = medium, 2 = high)
    pub quality: c_int,
}

impl Default for VoirsVoiceInfoDetailed {
    fn default() -> Self {
        Self {
            id: ptr::null_mut(),
            name: ptr::null_mut(),
            language: ptr::null_mut(),
            gender: 0,
            quality: 0,
        }
    }
}

/// Detailed voice list structure for C API
#[repr(C)]
#[derive(Debug)]
pub struct VoirsVoiceListDetailed {
    /// Array of voice information
    pub voices: *mut VoirsVoiceInfoDetailed,
    /// Number of voices in the array
    pub count: c_uint,
}

impl Default for VoirsVoiceListDetailed {
    fn default() -> Self {
        Self {
            voices: ptr::null_mut(),
            count: 0,
        }
    }
}

/// Set the active voice for a pipeline
///
/// # Arguments
/// * `pipeline_id` - Pipeline ID
/// * `voice_id` - Voice ID (null-terminated string)
///
/// Returns 0 on success, or error code on failure.
#[no_mangle]
pub extern "C" fn voirs_set_voice(pipeline_id: c_uint, voice_id: *const c_char) -> c_int {
    // `run_with_fallback_error` only installs the generic message below if
    // `set_voice_impl` didn't already set a more specific one (e.g. the
    // honest "this is a VOIRS_BENCHMARK_MODE placeholder" message from
    // crate::invalid_pipeline_message) -- so this wrapper never clobbers it.
    // It also can't mistake a stale message from a wholly unrelated earlier
    // call for "this call already reported something specific" the way a
    // plain "is any error pending" check could (set_voice_impl has
    // message-less `Err` paths, e.g. `pipeline_id == 0`) -- see
    // `run_with_fallback_error`'s doc comment for both failure modes.
    match crate::run_with_fallback_error(
        || set_voice_impl(pipeline_id, voice_id),
        |code| format!("Failed to set voice for pipeline {pipeline_id}: {code:?}"),
    ) {
        Ok(()) => 0,
        Err(code) => code as c_int,
    }
}

/// Get the current active voice for a pipeline
///
/// # Arguments
/// * `pipeline_id` - Pipeline ID
///
/// Returns a pointer to the voice ID string on success, or null on failure.
/// The returned string must be freed with `voirs_free_string()`.
#[no_mangle]
pub extern "C" fn voirs_get_voice(pipeline_id: c_uint) -> *mut c_char {
    // See voirs_set_voice's identical use of run_with_fallback_error and
    // rationale: don't clobber a more specific message get_voice_impl may
    // have already set, and don't mistake a stale message from an unrelated
    // earlier call for one either.
    let voice_id = match crate::run_with_fallback_error(
        || get_voice_impl(pipeline_id),
        |code| format!("Failed to get voice for pipeline {pipeline_id}: {code:?}"),
    ) {
        Ok(voice_id) => voice_id,
        Err(_) => return ptr::null_mut(),
    };

    match CString::new(voice_id) {
        Ok(c_str) => c_str.into_raw(),
        Err(e) => {
            set_last_error(format!("Failed to convert voice ID to C string: {e}"));
            ptr::null_mut()
        }
    }
}

/// List all available voices
///
/// Returns a pointer to a VoirsVoiceListDetailed structure on success, or null on failure.
/// The returned list must be freed with `voirs_free_voice_list()`.
#[no_mangle]
pub extern "C" fn voirs_list_voices() -> *mut VoirsVoiceListDetailed {
    match list_voices_impl() {
        Ok(voice_list) => Box::into_raw(Box::new(voice_list)),
        Err(code) => {
            set_last_error(format!("Failed to list voices: {code:?}"));
            ptr::null_mut()
        }
    }
}

/// Free a voice list structure
///
/// # Arguments
/// * `voice_list` - Voice list to free
///
/// # Safety
/// This function is unsafe because it deallocates raw memory.
/// The caller must ensure the voice list was allocated by this library.
#[no_mangle]
pub unsafe extern "C" fn voirs_free_voice_list(voice_list: *mut VoirsVoiceListDetailed) {
    if voice_list.is_null() {
        return;
    }

    let list = Box::from_raw(voice_list);

    // Free individual voice info structures
    if !list.voices.is_null() {
        let voices = std::slice::from_raw_parts_mut(list.voices, list.count as usize);
        for voice in voices {
            free_voice_info(voice);
        }
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            list.voices,
            list.count as usize,
        ));
    }
}

/// Free a voice info structure's strings
///
/// # Safety
/// This function is unsafe because it deallocates raw memory.
unsafe fn free_voice_info(voice: &mut VoirsVoiceInfoDetailed) {
    if !voice.id.is_null() {
        let _ = CString::from_raw(voice.id);
        voice.id = ptr::null_mut();
    }
    if !voice.name.is_null() {
        let _ = CString::from_raw(voice.name);
        voice.name = ptr::null_mut();
    }
    if !voice.language.is_null() {
        let _ = CString::from_raw(voice.language);
        voice.language = ptr::null_mut();
    }
}

/// Get information about a specific voice
///
/// # Arguments
/// * `voice_id` - Voice ID (null-terminated string)
///
/// Returns a pointer to a VoirsVoiceInfoDetailed structure on success, or null on failure.
/// The returned info must be freed with `voirs_free_voice_info()`.
#[no_mangle]
pub extern "C" fn voirs_get_voice_info(voice_id: *const c_char) -> *mut VoirsVoiceInfoDetailed {
    // Same non-clobbering/non-stale-leak guard as voirs_set_voice:
    // get_voice_info_impl sets a specific message for invalid UTF-8 (but not
    // for a null pointer or an unrecognized voice ID), which an unconditional
    // set_last_error here used to silently overwrite with the generic
    // "Failed to get voice info: InvalidParameter" for every failure alike.
    match crate::run_with_fallback_error(
        || get_voice_info_impl(voice_id),
        |code| format!("Failed to get voice info: {code:?}"),
    ) {
        Ok(voice_info) => Box::into_raw(Box::new(voice_info)),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a voice info structure
///
/// # Arguments
/// * `voice_info` - Voice info to free
///
/// # Safety
/// This function is unsafe because it deallocates raw memory.
#[no_mangle]
pub unsafe extern "C" fn voirs_free_voice_info(voice_info: *mut VoirsVoiceInfoDetailed) {
    if voice_info.is_null() {
        return;
    }

    let mut info = Box::from_raw(voice_info);
    free_voice_info(&mut info);
}

// Implementation functions

fn set_voice_impl(pipeline_id: c_uint, voice_id: *const c_char) -> Result<(), VoirsErrorCode> {
    if pipeline_id == 0 {
        return Err(VoirsErrorCode::InvalidParameter);
    }

    if voice_id.is_null() {
        return Err(VoirsErrorCode::InvalidParameter);
    }

    let voice_str = unsafe {
        match CStr::from_ptr(voice_id).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid UTF-8 in voice ID: {e}"));
                return Err(VoirsErrorCode::InvalidParameter);
            }
        }
    };

    #[cfg(feature = "ffi-test-mocks")]
    {
        // Mock mode: validate pipeline ID using the test tracking system
        use crate::c_api::core::{CREATED_PIPELINES, DESTROYED_PIPELINES};

        let created = match CREATED_PIPELINES.lock() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(VoirsErrorCode::InternalError);
            }
        };
        if !created.contains(&pipeline_id) {
            return Err(VoirsErrorCode::InvalidParameter);
        }
        drop(created);

        let destroyed = match DESTROYED_PIPELINES.lock() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(VoirsErrorCode::InternalError);
            }
        };
        if destroyed.contains(&pipeline_id) {
            return Err(VoirsErrorCode::InvalidParameter);
        }

        // Mock mode: just return success for valid pipeline IDs
        Ok(())
    }

    #[cfg(not(feature = "ffi-test-mocks"))]
    {
        let manager = get_pipeline_manager();
        let guard = manager.lock();
        let pipeline = match guard.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => {
                let is_placeholder = guard.is_placeholder(pipeline_id);
                drop(guard);
                set_last_error(crate::invalid_pipeline_message(pipeline_id, is_placeholder));
                return Err(VoirsErrorCode::InvalidParameter);
            }
        };
        drop(guard);

        let runtime = get_runtime()?;

        runtime
            .block_on(async { pipeline.set_voice(voice_str).await })
            .map_err(|e| {
                set_last_error(format!("Failed to set voice '{voice_str}': {e}"));
                VoirsErrorCode::VoiceNotFound
            })?;

        Ok(())
    }
}

fn get_voice_impl(pipeline_id: c_uint) -> Result<String, VoirsErrorCode> {
    if pipeline_id == 0 {
        return Err(VoirsErrorCode::InvalidParameter);
    }

    #[cfg(feature = "ffi-test-mocks")]
    {
        // Mock mode: validate pipeline ID using the test tracking system
        use crate::c_api::core::{CREATED_PIPELINES, DESTROYED_PIPELINES};

        let created = match CREATED_PIPELINES.lock() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(VoirsErrorCode::InternalError);
            }
        };
        if !created.contains(&pipeline_id) {
            return Err(VoirsErrorCode::InvalidParameter);
        }
        drop(created);

        let destroyed = match DESTROYED_PIPELINES.lock() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(VoirsErrorCode::InternalError);
            }
        };
        if destroyed.contains(&pipeline_id) {
            return Err(VoirsErrorCode::InvalidParameter);
        }

        // Mock mode: just return a default voice ID
        Ok("default".to_string())
    }

    #[cfg(not(feature = "ffi-test-mocks"))]
    {
        let manager = get_pipeline_manager();
        let guard = manager.lock();
        let pipeline = match guard.get_pipeline(pipeline_id) {
            Some(p) => p,
            None => {
                let is_placeholder = guard.is_placeholder(pipeline_id);
                drop(guard);
                set_last_error(crate::invalid_pipeline_message(pipeline_id, is_placeholder));
                return Err(VoirsErrorCode::InvalidParameter);
            }
        };
        drop(guard);

        let runtime = get_runtime()?;

        let voice_config = runtime.block_on(async { pipeline.current_voice().await });

        let voice_id = voice_config
            .map(|config| config.id)
            .unwrap_or_else(|| "default".to_string());

        Ok(voice_id)
    }
}

fn list_voices_impl() -> Result<VoirsVoiceListDetailed, VoirsErrorCode> {
    let runtime = get_runtime()?;

    let voices = runtime.block_on(async {
        // In a real implementation, this would query the available voices
        // For now, we'll return a mock list
        vec![
            ("default", "Default Voice", "en-US", 0, 1),
            ("female", "Female Voice", "en-US", 2, 1),
            ("male", "Male Voice", "en-US", 1, 1),
        ]
    });

    let mut voice_infos = Vec::with_capacity(voices.len());

    for (id, name, lang, gender, quality) in voices {
        let voice_info = VoirsVoiceInfoDetailed {
            id: CString::new(id)
                .map_err(|_| VoirsErrorCode::InternalError)?
                .into_raw(),
            name: CString::new(name)
                .map_err(|_| VoirsErrorCode::InternalError)?
                .into_raw(),
            language: CString::new(lang)
                .map_err(|_| VoirsErrorCode::InternalError)?
                .into_raw(),
            gender,
            quality,
        };
        voice_infos.push(voice_info);
    }

    // `.into_boxed_slice()` guarantees capacity == length, so the pointer +
    // length pair below is reconstructible via `Box::from_raw` in
    // `voirs_free_voice_list` regardless of how many entries are ever pushed
    // (unlike relying on `Vec::with_capacity(voices.len())` happening to
    // equal the final length -- see `VoirsAudioBuffer::from_audio_buffer`
    // for the same pattern used elsewhere in this crate).
    let mut voice_infos = voice_infos.into_boxed_slice();
    let voice_list = VoirsVoiceListDetailed {
        voices: voice_infos.as_mut_ptr(),
        count: voice_infos.len() as c_uint,
    };

    // Prevent the boxed slice from being dropped; ownership transfers to the
    // raw pointer above, reclaimed by `Box::from_raw` in `voirs_free_voice_list`.
    std::mem::forget(voice_infos);

    Ok(voice_list)
}

fn get_voice_info_impl(voice_id: *const c_char) -> Result<VoirsVoiceInfoDetailed, VoirsErrorCode> {
    if voice_id.is_null() {
        return Err(VoirsErrorCode::InvalidParameter);
    }

    let voice_str = unsafe {
        match CStr::from_ptr(voice_id).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid UTF-8 in voice ID: {e}"));
                return Err(VoirsErrorCode::InvalidParameter);
            }
        }
    };

    // In a real implementation, this would query the voice database
    // For now, we'll return mock data
    let (name, lang, gender, quality) = match voice_str {
        "default" => ("Default Voice", "en-US", 0, 1),
        "female" => ("Female Voice", "en-US", 2, 1),
        "male" => ("Male Voice", "en-US", 1, 1),
        _ => return Err(VoirsErrorCode::VoiceNotFound),
    };

    Ok(VoirsVoiceInfoDetailed {
        id: CString::new(voice_str)
            .map_err(|_| VoirsErrorCode::InternalError)?
            .into_raw(),
        name: CString::new(name)
            .map_err(|_| VoirsErrorCode::InternalError)?
            .into_raw(),
        language: CString::new(lang)
            .map_err(|_| VoirsErrorCode::InternalError)?
            .into_raw(),
        gender,
        quality,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_api::core::*;

    // This test creates a pipeline and expects instant, network-free success
    // -- true only for the mock pipeline bookkeeping in c_api::core, which is
    // gated behind the (off-by-default) `ffi-test-mocks` feature. The real
    // (mocks-off) pipeline creation + voice path is exercised by
    // `voirs-ffi/tests/pipeline_real_path.rs`.
    #[cfg(feature = "ffi-test-mocks")]
    #[test]
    fn test_voice_operations() {
        // Create a pipeline first
        let pipeline_id = voirs_create_pipeline();
        assert_ne!(pipeline_id, 0, "Pipeline creation should succeed");

        // Test setting voice
        let voice_id = std::ffi::CString::new("default").unwrap();
        let result = voirs_set_voice(pipeline_id, voice_id.as_ptr());
        assert_eq!(result, 0, "Setting voice should succeed");

        // Test getting voice
        let current_voice = voirs_get_voice(pipeline_id);
        assert!(!current_voice.is_null(), "Getting voice should succeed");

        // Free the voice string
        unsafe {
            if !current_voice.is_null() {
                let _ = CString::from_raw(current_voice);
            }
        }

        // Clean up
        let result = voirs_destroy_pipeline(pipeline_id);
        assert_eq!(result, 0, "Pipeline destruction should succeed");
    }

    #[test]
    fn test_voice_list() {
        // Test listing voices
        let voice_list = voirs_list_voices();
        assert!(!voice_list.is_null(), "Listing voices should succeed");

        unsafe {
            let list = &*voice_list;
            assert!(list.count > 0, "Voice list should contain voices");
            assert!(!list.voices.is_null(), "Voice array should not be null");

            // Test voice info
            let voices = std::slice::from_raw_parts(list.voices, list.count as usize);
            for voice in voices {
                assert!(!voice.id.is_null(), "Voice ID should not be null");
                assert!(!voice.name.is_null(), "Voice name should not be null");
                assert!(
                    !voice.language.is_null(),
                    "Voice language should not be null"
                );
            }

            // Free the voice list
            voirs_free_voice_list(voice_list);
        }
    }

    #[test]
    fn test_voice_info() {
        let voice_id = std::ffi::CString::new("default").unwrap();
        let voice_info = voirs_get_voice_info(voice_id.as_ptr());
        assert!(!voice_info.is_null(), "Getting voice info should succeed");

        unsafe {
            let info = &*voice_info;
            assert!(!info.id.is_null(), "Voice ID should not be null");
            assert!(!info.name.is_null(), "Voice name should not be null");
            assert!(
                !info.language.is_null(),
                "Voice language should not be null"
            );

            // Free the voice info
            voirs_free_voice_info(voice_info);
        }
    }

    #[test]
    fn test_invalid_voice_operations() {
        // Test with invalid pipeline ID
        let voice_id = std::ffi::CString::new("default").unwrap();
        let result = voirs_set_voice(0, voice_id.as_ptr());
        assert_ne!(result, 0, "Setting voice with invalid pipeline should fail");

        // Test with null voice ID
        let result = voirs_set_voice(1, std::ptr::null());
        assert_ne!(result, 0, "Setting null voice should fail");

        // Test getting voice for invalid pipeline
        let current_voice = voirs_get_voice(0);
        assert!(
            current_voice.is_null(),
            "Getting voice for invalid pipeline should fail"
        );

        // Test getting info for invalid voice
        let invalid_voice = std::ffi::CString::new("nonexistent").unwrap();
        let voice_info = voirs_get_voice_info(invalid_voice.as_ptr());
        assert!(
            voice_info.is_null(),
            "Getting info for invalid voice should fail"
        );
    }

    /// Regression test for a stale-error-leak bug: `set_voice_impl`'s
    /// `pipeline_id == 0` check returns `Err` without calling
    /// `set_last_error` (there's nothing pipeline-specific to say about ID
    /// `0`). A naive "is an error already pending" check would mistake a
    /// message left over from a completely unrelated EARLIER call on this
    /// thread for "this call already reported something specific," silently
    /// leaving the stale text in place instead of reporting this call's own
    /// outcome. `run_with_fallback_error` (see its doc comment in `lib.rs`)
    /// avoids this by comparing the message immediately before/after
    /// `set_voice_impl`/`get_voice_impl` run rather than checking "any error
    /// pending." Deterministic and feature-independent: the
    /// `pipeline_id == 0` check happens before any `ffi-test-mocks`/
    /// production split.
    #[test]
    fn test_set_and_get_voice_do_not_leak_stale_error_from_earlier_call() {
        let stale = "stale message from a totally unrelated earlier call";

        crate::voirs_clear_error();
        crate::set_last_error(stale.to_string());
        let voice_id = std::ffi::CString::new("default").unwrap();
        let result = voirs_set_voice(0, voice_id.as_ptr());
        assert_ne!(result, 0);
        let message = unsafe {
            let ptr = crate::voirs_get_last_error();
            assert!(!ptr.is_null());
            let s = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
            crate::voirs_free_string(ptr);
            s
        };
        assert!(
            !message.contains(stale),
            "voirs_set_voice must not leak a stale error from an unrelated \
             earlier call: {message}"
        );

        crate::voirs_clear_error();
        crate::set_last_error(stale.to_string());
        let voice_ptr = voirs_get_voice(0);
        assert!(voice_ptr.is_null());
        let message = unsafe {
            let ptr = crate::voirs_get_last_error();
            assert!(!ptr.is_null());
            let s = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
            crate::voirs_free_string(ptr);
            s
        };
        assert!(
            !message.contains(stale),
            "voirs_get_voice must not leak a stale error from an unrelated \
             earlier call: {message}"
        );
    }

    /// Regression test for `voirs_get_voice_info` specifically: unlike
    /// `voirs_set_voice`/`voirs_get_voice` (fixed earlier), this wrapper
    /// still called `set_last_error` unconditionally, so
    /// `get_voice_info_impl`'s specific "Invalid UTF-8 in voice ID: ..."
    /// message (set when `voice_id` isn't valid UTF-8) was silently
    /// discarded in favor of the generic
    /// "Failed to get voice info: InvalidParameter". Deterministic and
    /// feature-independent: UTF-8 validation happens unconditionally, before
    /// any pipeline/runtime access.
    #[test]
    fn test_get_voice_info_invalid_utf8_error_is_not_clobbered() {
        crate::voirs_clear_error();

        let invalid_utf8 = std::ffi::CString::new(vec![0xFFu8, 0xFEu8]).expect("no interior NUL");
        let info = voirs_get_voice_info(invalid_utf8.as_ptr());
        assert!(info.is_null());

        let message = unsafe {
            let ptr = crate::voirs_get_last_error();
            assert!(!ptr.is_null());
            let s = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
            crate::voirs_free_string(ptr);
            s
        };
        assert!(
            message.contains("Invalid UTF-8"),
            "the specific inner diagnostic must survive to the caller, got: {message}"
        );
        assert!(
            !message.contains("Failed to get voice info: InvalidParameter"),
            "must not be clobbered by the generic VoirsErrorCode-only \
             fallback: {message}"
        );
    }

    /// A *successful* `voirs_get_voice_info`/`voirs_set_voice` call must
    /// never clear or touch a message still pending from an earlier,
    /// unrelated failed call -- `run_with_fallback_error` only ever installs
    /// a message in response to its own `f`'s `Err`, matching the existing
    /// sticky-until-explicitly-cleared-or-overwritten contract
    /// `voirs_clear_error()`'s existence as a distinct public API implies.
    /// (This is the property an earlier, since-reverted fix -- clearing the
    /// error unconditionally at each wrapper's entry -- would have violated.)
    #[test]
    fn test_successful_get_voice_info_leaves_unrelated_pending_message_untouched() {
        crate::voirs_clear_error();
        crate::set_last_error("earlier unrelated failure".to_string());

        let default_voice = std::ffi::CString::new("default").unwrap();
        let info = voirs_get_voice_info(default_voice.as_ptr());
        assert!(!info.is_null(), "\"default\" is a recognized voice ID");
        unsafe {
            voirs_free_voice_info(info);
        }

        let message = unsafe {
            let ptr = crate::voirs_get_last_error();
            assert!(!ptr.is_null());
            let s = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
            crate::voirs_free_string(ptr);
            s
        };
        assert_eq!(
            message, "earlier unrelated failure",
            "a successful call must not disturb an unrelated pending message"
        );
    }
}
