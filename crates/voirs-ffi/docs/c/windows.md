# Windows Integration Guide

This page documents the real, currently-exported C API of the `voirs-ffi` crate (library name
`voirs`) as built for Windows, and how to build/link against it. It replaces an earlier revision
that documented roughly 53 functions, of which only about 8 actually existed (COM device
enumeration, WASAPI exclusive mode, and a `VoirsWindowsPerformanceProfiler` API never existed in
this crate); every symbol listed below was verified against `crates/voirs-ffi/src/**/*.rs`
directly.

No C header ships with this crate (there is no `cbindgen` build step), so every prototype below is
what a caller must declare itself.

## Building

```powershell
# From the workspace root
cargo build --release -p voirs-ffi
```

This produces `target\release\voirs.dll` plus the import library `target\release\voirs.dll.lib`
(crate `[lib] name = "voirs"`, `crate-type = ["cdylib", "rlib"]` — dynamic library only, no static
`.lib`-only build).

**Build note:** `crates/voirs-ffi/src/platform/windows.rs` unconditionally imports the `winapi`
crate whenever `target_os = "windows"`, but `winapi` (and the `windows` crate) are only pulled in
as dependencies by the optional `windows-platform` Cargo feature
(`crates/voirs-ffi/Cargo.toml`). Build with that feature enabled on Windows:

```powershell
cargo build --release -p voirs-ffi --features windows-platform
```

## Linking

### MSVC

```cmd
cl myapp.c /I. /link /LIBPATH:target\release voirs.dll.lib
```

Copy `voirs.dll` next to `myapp.exe`, or ensure it is on `PATH`, before running.

### MinGW

```bash
gcc myapp.c -I. -Ltarget/release -lvoirs -o myapp.exe
```

Unlike Linux/macOS, there is no environment-variable equivalent of `LD_LIBRARY_PATH`/
`DYLD_LIBRARY_PATH` that Windows honors for arbitrary DLL search directories by default — the
loader searches (in order) the application directory, system directories, and `PATH`. Placing
`voirs.dll` next to the `.exe`, or adding its directory to `PATH`, are the two practical options.

## Minimal example

See the [crate README's Quick Start](../../README.md#quick-start) for a complete, compilable
`voirs_synthesize_advanced` example (it declares the small subset of the real ABI it uses, since no
header ships). Nothing in it is Windows-specific; it builds and links the same way as shown above.

## Windows-specific functions (`src/platform/windows.rs`)

These 5 functions only compile when `target_os = "windows"` (`crates/voirs-ffi/src/platform/mod.rs`
gates `pub mod windows;` on it), and internally use real WASAPI/COM calls (`winapi` crate) rather
than placeholders:

```c
/* Opaque handle -- only ever used through the functions below. */
typedef struct WindowsAudioSession WindowsAudioSession;

WindowsAudioSession *voirs_windows_init_audio_session(void);
void voirs_windows_destroy_audio_session(WindowsAudioSession *session);

float voirs_windows_get_system_volume(WindowsAudioSession *session);
bool  voirs_windows_set_system_volume(WindowsAudioSession *session, float volume);

/* Caller must free the returned string with voirs_free_string(). Returns NULL on failure
 * (including when key_name is NULL). */
char *voirs_windows_read_registry_config(const char *key_name);
```

`voirs_windows_init_audio_session` calls `CoInitialize` and `CoCreateInstance` for
`MMDeviceEnumerator` internally (real COM usage, not a placeholder); `voirs_windows_get_system_volume`/
`voirs_windows_set_system_volume` go through `IAudioEndpointVolume`.

## Cross-platform functions (identical on Linux/macOS/Windows)

Everything below compiles regardless of target OS. Full signatures live in the source files named;
this is a complete, machine-checked name index grouped by module (not padded, not invented):

**Pipeline lifecycle** (`src/c_api/core.rs`, 6 functions)

`voirs_create_pipeline`, `voirs_create_pipeline_with_config`, `voirs_destroy_pipeline`, `voirs_get_pipeline_count`, `voirs_is_pipeline_benchmark_placeholder`, `voirs_is_pipeline_valid`

**Synthesis** (`src/c_api/synthesis.rs`, 10 functions)

`voirs_free_batch_synthesis_result`, `voirs_free_synthesis_result`, `voirs_get_synthesis_stats`, `voirs_reset_synthesis_stats`, `voirs_synthesize_advanced`, `voirs_synthesize_batch`, `voirs_synthesize_batch_advanced`, `voirs_synthesize_streaming`, `voirs_synthesize_streaming_advanced`, `voirs_synthesize_streaming_realtime`

**Voice management** (`src/c_api/voice.rs`, 6 functions)

`voirs_free_voice_info`, `voirs_free_voice_list`, `voirs_get_voice`, `voirs_get_voice_info`, `voirs_list_voices`, `voirs_set_voice`

**Audio post-processing / effects (extended)** (`src/c_api/audio.rs`, 8 functions)

`voirs_audio_apply_effects`, `voirs_audio_crossfade`, `voirs_audio_duplicate`, `voirs_audio_get_statistics`, `voirs_audio_get_supported_formats`, `voirs_audio_mix`, `voirs_audio_save_flac`, `voirs_audio_save_mp3`

**Audio analysis & DSP utilities** (`src/utils/audio.rs`, 27 functions)

`voirs_audio_analyze`, `voirs_audio_apply_compression`, `voirs_audio_apply_multiband_eq`, `voirs_audio_calculate_brightness`, `voirs_audio_calculate_hnr`, `voirs_audio_calculate_spectral_flux`, `voirs_audio_calculate_spectral_rolloff`, `voirs_audio_enhance_quality`, `voirs_audio_fade_in`, `voirs_audio_fade_out`, `voirs_audio_get_peak`, `voirs_audio_get_rms`, `voirs_audio_low_pass_filter`, `voirs_audio_normalize`, `voirs_audio_remove_dc`, `voirs_audio_soft_limiter`, `voirs_performance_monitor_create`, `voirs_performance_monitor_free`, `voirs_performance_monitor_get_summary`, `voirs_performance_monitor_record_audio_time`, `voirs_performance_monitor_record_cpu`, `voirs_performance_monitor_record_memory`, `voirs_regression_detector_add_measurement`, `voirs_regression_detector_check`, `voirs_regression_detector_create`, `voirs_regression_detector_free`, `voirs_regression_detector_set_baseline`

**Sample format / rate conversion** (`src/c_api/convert.rs`, 17 functions)

`voirs_audio_convert_format`, `voirs_convert_double_to_float`, `voirs_convert_float_to_double`, `voirs_convert_float_to_int16`, `voirs_convert_float_to_int24`, `voirs_convert_float_to_int32`, `voirs_convert_float_to_uint16`, `voirs_convert_float_to_uint32`, `voirs_convert_float_to_uint8`, `voirs_convert_int16_to_float`, `voirs_convert_int24_to_float`, `voirs_convert_mono_to_stereo`, `voirs_convert_sample_rate`, `voirs_convert_stereo_to_mono`, `voirs_convert_uint16_to_float`, `voirs_convert_uint32_to_float`, `voirs_convert_uint8_to_float`

**Configuration builders & validation** (`src/c_api/config.rs`, 8 functions)

`voirs_config_apply_synthesis_preset`, `voirs_config_create_model_default`, `voirs_config_create_performance_default`, `voirs_config_create_synthesis_default`, `voirs_config_get_synthesis_info`, `voirs_config_validate_model`, `voirs_config_validate_performance`, `voirs_config_validate_synthesis`

**Threading & async/callback synthesis** (`src/c_api/threading.rs`, 13 functions)

`voirs_cancel_synthesis`, `voirs_get_active_operations`, `voirs_get_global_thread_count`, `voirs_get_max_concurrent`, `voirs_get_thread_stats`, `voirs_is_thread_pool_enabled`, `voirs_register_callbacks`, `voirs_set_global_thread_count`, `voirs_set_max_concurrent`, `voirs_set_thread_pool_enabled`, `voirs_synthesize_async`, `voirs_synthesize_parallel`, `voirs_unregister_callbacks`

**Custom allocator control** (`src/c_api/allocator.rs`, 6 functions)

`voirs_get_allocator_name`, `voirs_get_allocator_stats`, `voirs_get_memory_fragmentation`, `voirs_has_custom_allocator`, `voirs_reset_allocator_stats`, `voirs_set_allocator`

**Zero-copy buffers, ring buffers, memory-mapped files** (`src/c_api/zero_copy.rs`, 32 functions)

`voirs_memory_map_advise_random`, `voirs_memory_map_advise_sequential`, `voirs_memory_map_data`, `voirs_memory_map_data_mut`, `voirs_memory_map_destroy`, `voirs_memory_map_open_read`, `voirs_memory_map_open_write`, `voirs_memory_map_size`, `voirs_memory_map_sync`, `voirs_zero_copy_batch_copy`, `voirs_zero_copy_buffer_capacity`, `voirs_zero_copy_buffer_clone`, `voirs_zero_copy_buffer_create`, `voirs_zero_copy_buffer_data`, `voirs_zero_copy_buffer_data_mut`, `voirs_zero_copy_buffer_destroy`, `voirs_zero_copy_buffer_len`, `voirs_zero_copy_buffer_ref_count`, `voirs_zero_copy_buffer_set_len`, `voirs_zero_copy_buffer_slice`, `voirs_zero_copy_deinterleave`, `voirs_zero_copy_interleave`, `voirs_zero_copy_ring_available_read`, `voirs_zero_copy_ring_available_write`, `voirs_zero_copy_ring_capacity`, `voirs_zero_copy_ring_create`, `voirs_zero_copy_ring_destroy`, `voirs_zero_copy_ring_read`, `voirs_zero_copy_ring_write`, `voirs_zero_copy_view_data`, `voirs_zero_copy_view_destroy`, `voirs_zero_copy_view_len`

**Misc utilities (version, logging, validation)** (`src/c_api/utils.rs`, 16 functions)

`voirs_calculate_aligned_size`, `voirs_get_build_info`, `voirs_get_error_description`, `voirs_get_memory_stats`, `voirs_get_process_memory_usage`, `voirs_get_recommended_buffer_size`, `voirs_get_system_info`, `voirs_get_version_string`, `voirs_is_log_level_enabled`, `voirs_log_message`, `voirs_reset_memory_stats`, `voirs_set_log_callback`, `voirs_validate_audio_format`, `voirs_validate_buffer`, `voirs_validate_range_float`, `voirs_validate_range_uint`

**Memory statistics (buffer tracking)** (`src/memory.rs`, 4 functions)

`voirs_memory_check_leaks`, `voirs_memory_clear_pools`, `voirs_memory_get_stats`, `voirs_memory_reset_stats`

**Performance / SIMD / batch helpers** (`src/performance.rs`, 6 functions)

`voirs_batch_convert_format`, `voirs_batch_process_audio`, `voirs_convert_f32_to_i16_optimized`, `voirs_detect_cpu_features`, `voirs_get_optimal_performance_config`, `voirs_interleave_audio_optimized`

**Batch-config helpers** (`src/utils/batch_ops.rs`, 6 functions)

`voirs_batch_config_create`, `voirs_batch_config_free`, `voirs_batch_config_set_cache`, `voirs_batch_config_set_sample_rate`, `voirs_batch_config_set_voice`, `voirs_batch_config_set_workers`

**Error message localization** (`src/error/i18n.rs`, 3 functions)

`voirs_get_locale`, `voirs_get_localized_message`, `voirs_set_locale`

**Structured error aggregation** (`src/error/structured.rs`, 3 functions)

`voirs_clear_error_aggregator`, `voirs_get_error_stats`, `voirs_get_recent_errors`

**Error recovery hints** (`src/error/recovery.rs`, 2 functions)

`voirs_attempt_recovery`, `voirs_get_recovery_stats`

**Generic platform info** (`src/platform/mod.rs`, 6 functions)

`voirs_get_audio_config_low_latency`, `voirs_get_audio_config_optimal`, `voirs_get_optimal_buffer_size`, `voirs_get_optimal_threads`, `voirs_get_platform_info`, `voirs_supports_hardware_acceleration`

**Linux package building (.deb) — compiled on every OS this crate builds for** (`src/platform/packages.rs`, 4 functions)

`voirs_package_build_all`, `voirs_package_build_debian`, `voirs_package_create_manager`, `voirs_package_destroy_manager`

**Visual Studio project integration** (`src/platform/vs.rs`, 5 functions) — compiled on every OS this crate builds for, but naturally most relevant here:

`voirs_vs_create_integration`, `voirs_vs_destroy_integration`, `voirs_vs_get_version`, `voirs_vs_install_integration`, `voirs_vs_verify_installation`

**Xcode project integration — compiled on every OS this crate builds for** (`src/platform/xcode.rs`, 5 functions)

`voirs_xcode_build_framework`, `voirs_xcode_create_integration`, `voirs_xcode_destroy_integration`, `voirs_xcode_install_integration`, `voirs_xcode_verify_installation`

**Core error/string/buffer plumbing** (`src/lib.rs`, 6 functions)

`voirs_clear_error`, `voirs_error_message`, `voirs_free_audio_buffer`, `voirs_free_string`, `voirs_get_last_error`, `voirs_has_error`

*(cross-platform total: 199 functions, identical on Linux/macOS/Windows)*

Full prototypes for the core subset used in the Quick Start example (pipeline lifecycle, synthesis,
voice management, error handling) are in the [crate README](../../README.md).

## Visual Studio project integration

`voirs_vs_create_integration`, `voirs_vs_install_integration`, `voirs_vs_verify_installation`,
`voirs_vs_get_version`, and `voirs_vs_destroy_integration` (`src/platform/vs.rs`) are real
dev-tooling entry points (MSBuild target files, IntelliSense configuration generation for this
crate's own build), not something a typical embedding application calls at synthesis time.

## Troubleshooting

- **`The code execution cannot proceed because voirs.dll was not found`** — the loader can't find
  the DLL; see [Linking](#linking) above (place `voirs.dll` next to the `.exe` or add it to `PATH`).
- **Build fails with unresolved `winapi::...` imports** — build with
  `--features windows-platform` (see the build note above).
- **`undefined symbol: voirs_synthesize`** — that function does not exist. The one-shot
  self-contained entry point is `voirs_synthesize_advanced`; the streaming entry points are
  `voirs_synthesize_streaming`, `voirs_synthesize_streaming_advanced`, and
  `voirs_synthesize_streaming_realtime`. See [C API overview](../../README.md#c-api-overview) in
  the crate README.
- **Synthesis returns `VOIRS_ERROR_SYNTHESIS_FAILED`/`VOIRS_ERROR_INITIALIZATION_FAILED`** — real
  model weights could not be loaded (e.g. no network access on first run). Call
  `voirs_get_last_error()` for the underlying `voirs_sdk` error message.
