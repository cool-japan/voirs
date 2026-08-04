# voirs

[![Crates.io](https://img.shields.io/crates/v/voirs-ffi.svg)](https://crates.io/crates/voirs-ffi)
[![Documentation](https://docs.rs/voirs-ffi/badge.svg)](https://docs.rs/voirs-ffi)

**Foreign Function Interface (FFI) bindings for VoiRS speech synthesis.**

This crate builds a single native library (Cargo package `voirs-ffi`, library name **`voirs`**)
that exposes the VoiRS pipeline to other languages: a C-compatible ABI (always built), and
optional Python (PyO3) and Node.js (N-API) bindings behind Cargo feature flags.

> **Note on this document:** every function and struct named below was verified against the
> current source under `src/` at the time of writing. This crate does **not** ship a generated
> C header (no `cbindgen` build step exists yet), so a C caller must declare the prototypes it
> uses itself, as the example below does.

## Bindings

| Binding | Mechanism | How to enable | Status |
|---|---|---|---|
| **C / C++** | raw `extern "C"` functions in `src/c_api/*.rs`, `src/lib.rs`, `src/memory.rs`, `src/performance.rs`, `src/platform/*.rs`, `src/error/*.rs`, `src/utils/*.rs` | built by default (no feature flag) | Present, no header shipped |
| **Python** | PyO3 module `voirs` (`src/python/`) | `--features python` (add `numpy` for `PyAudioBuffer::as_numpy`) | Present |
| **Node.js** | N-API bindings (`src/nodejs.rs`) | `--features nodejs` | Present |
| **WebAssembly** | `wasm-bindgen` bindings (`src/wasm.rs`) | `--features wasm` | Present |

Other feature flags worth knowing about:

- `gpu` — forwards to `voirs-acoustic`/`voirs-vocoder` GPU support.
- `codecs` — forwards to `voirs-vocoder/ffi-codecs`; without it, `voirs_audio_save_mp3` compiles
  and links but returns `VOIRS_ERROR_INTERNAL_ERROR` instead of encoding. `voirs_audio_save_flac`
  works either way.
- `linux-platform` / `macos-platform` / `windows-platform` — pull in ALSA/PulseAudio/D-Bus, cpal,
  and `windows`-crate dependencies respectively, so the platform-specific functions described in
  [`docs/c/linux.md`](docs/c/linux.md), [`docs/c/macos.md`](docs/c/macos.md), and
  [`docs/c/windows.md`](docs/c/windows.md) do real device/system queries instead of falling back
  to conservative defaults.
- `ffi-test-mocks` — off by default; used only by this crate's own test suite to swap the real
  pipeline manager for an in-memory fake. Never enable this in an application.

## Quick Start

### C

`voirs_synthesize_advanced()` builds and internally caches its own pipeline, so a minimal example
does not need to call `voirs_create_pipeline()` at all — see
["Pipeline handles vs. self-contained synthesis"](#pipeline-handles-vs-self-contained-synthesis)
below for when you *do* want an explicit pipeline handle (e.g. to call `voirs_set_voice`).

```c
/* quickstart.c
 *
 * No C header ships with voirs-ffi yet, so we declare the small slice of the
 * real #[repr(C)] ABI this example uses. Types/signatures must match
 * crates/voirs-ffi/src/lib.rs and crates/voirs-ffi/src/c_api/synthesis.rs.
 */
#include <stdio.h>
#include <stdbool.h>

typedef enum {
    VOIRS_SUCCESS = 0,
    VOIRS_ERROR_INVALID_PARAMETER = 1,
    VOIRS_ERROR_INITIALIZATION_FAILED = 2,
    VOIRS_ERROR_SYNTHESIS_FAILED = 3,
    VOIRS_ERROR_VOICE_NOT_FOUND = 4,
    VOIRS_ERROR_IO_ERROR = 5,
    VOIRS_ERROR_OUT_OF_MEMORY = 6,
    VOIRS_ERROR_OPERATION_CANCELLED = 7,
    VOIRS_ERROR_INTERNAL_ERROR = 99,
} VoirsErrorCode;

typedef struct {
    float *samples;
    unsigned int length;
    unsigned int sample_rate;
    unsigned int channels;
    float duration;
} VoirsAudioBuffer;

typedef enum { VOIRS_FORMAT_WAV = 0, VOIRS_FORMAT_FLAC = 1, VOIRS_FORMAT_MP3 = 2,
               VOIRS_FORMAT_OPUS = 3, VOIRS_FORMAT_OGG = 4 } VoirsAudioFormat;
typedef enum { VOIRS_QUALITY_LOW = 0, VOIRS_QUALITY_MEDIUM = 1,
               VOIRS_QUALITY_HIGH = 2, VOIRS_QUALITY_ULTRA = 3 } VoirsQualityLevel;

typedef struct {
    float speaking_rate;
    float pitch_shift;
    float volume_gain;
    int enable_enhancement;       /* 0/1 */
    VoirsAudioFormat output_format;
    unsigned int sample_rate;
    VoirsQualityLevel quality;
} VoirsSynthesisConfig;

typedef struct {
    VoirsSynthesisConfig base_config;
    bool enable_quality_analysis;
    bool enable_real_time_processing;
    bool enable_noise_reduction;
    bool enable_normalization;
    float target_loudness_lufs;
    unsigned int chunk_size_ms;
} VoirsAdvancedSynthesisConfig;

typedef struct {
    VoirsAudioBuffer *audio;
    float synthesis_time_ms;
    float quality_score;
    char *processing_info;
} VoirsSynthesisResult;

extern VoirsErrorCode voirs_synthesize_advanced(const char *text,
                                                 const VoirsAdvancedSynthesisConfig *config,
                                                 VoirsSynthesisResult *result);
extern void voirs_free_synthesis_result(VoirsSynthesisResult *result);
extern VoirsErrorCode voirs_audio_save_flac(const VoirsAudioBuffer *buffer,
                                             const char *filename,
                                             unsigned int compression_level);
extern char *voirs_get_last_error(void);
extern void voirs_free_string(char *s);
extern const char *voirs_error_message(VoirsErrorCode code);

int main(void) {
    VoirsSynthesisResult result;

    /* NULL config -> VoirsAdvancedSynthesisConfig::default() */
    VoirsErrorCode code = voirs_synthesize_advanced("Hello, world!", NULL, &result);
    if (code != VOIRS_SUCCESS) {
        char *detail = voirs_get_last_error();
        fprintf(stderr, "synthesis failed (%s): %s\n",
                voirs_error_message(code), detail ? detail : "(no detail)");
        if (detail) voirs_free_string(detail);
        return 1;
    }

    printf("synthesized %u samples @ %u Hz, %u channel(s), %.2f s\n",
           result.audio->length, result.audio->sample_rate,
           result.audio->channels, result.audio->duration);

    voirs_audio_save_flac(result.audio, "output.flac", /*compression_level=*/5);

    voirs_free_synthesis_result(&result); /* also frees result.audio and processing_info */
    return 0;
}
```

Build the library, then compile and link against it (see [Building](#building) for details):

```bash
cargo build --release -p voirs-ffi
gcc quickstart.c -Ltarget/release -lvoirs -o quickstart
LD_LIBRARY_PATH=target/release ./quickstart   # macOS: DYLD_LIBRARY_PATH instead
```

Real synthesis loads acoustic/vocoder model weights (first call may fetch them and can take a
while); if no models are reachable, `voirs_synthesize_advanced` returns
`VOIRS_ERROR_SYNTHESIS_FAILED` / `VOIRS_ERROR_INITIALIZATION_FAILED` rather than fabricating
audio — check `voirs_get_last_error()`.

### Python (`--features python`)

```python
import voirs

pipeline = voirs.VoirsPipeline()          # synchronous constructor, no asyncio needed
audio = pipeline.synthesize("Hello, world!")   # -> PyAudioBuffer

print(audio.sample_rate(), audio.channels(), audio.duration())  # methods, not properties
audio.save("output.wav")                  # format inferred from extension; "wav" is the default

# With --features numpy:
# samples = audio.as_numpy()
```

`VoirsPipeline.with_config(use_gpu=None, num_threads=None, cache_dir=None, device=None)` is a
`@staticmethod` alternative constructor. See `voirs.pyi` in this crate for the full stub, and
`src/python/pipeline.rs` / `src/python/audio_buffer.rs` for the implementation.

## Pipeline handles vs. self-contained synthesis

Two independent ways to drive synthesis exist in the C API; pick one per call site:

1. **Self-contained** (`voirs_synthesize_advanced`, `voirs_synthesize_streaming*`, `voirs_synthesize_batch*`) —
   builds/reuses an internally cached pipeline; no handle to manage.
2. **Explicit pipeline handle** — create one with `voirs_create_pipeline()` (returns a `uint32_t`
   ID, 0 on failure; check `voirs_get_last_error()`), then pass that ID to
   `voirs_set_voice(pipeline_id, voice_id)`, `voirs_get_voice(pipeline_id)`,
   `voirs_synthesize_async(...)`, or `voirs_synthesize_parallel(...)`; release it with
   `voirs_destroy_pipeline(pipeline_id)`. Validate with `voirs_is_pipeline_valid(pipeline_id)`.

```c
extern unsigned int voirs_create_pipeline(void);
extern int voirs_set_voice(unsigned int pipeline_id, const char *voice_id);
extern int voirs_destroy_pipeline(unsigned int pipeline_id);
```

Setting the environment variable `VOIRS_BENCHMARK_MODE=1` makes `voirs_create_pipeline()` return a
lightweight placeholder handle (skips model loading) for measuring handle-management overhead in
isolation; a placeholder handle fails any real synthesis/voice call. See the doc comment on
`voirs_create_pipeline` in `src/c_api/core.rs` for the full contract.

## Error handling

Errors are thread-local, not per-handle:

```c
extern int voirs_has_error(void);          /* 1 if a message is pending */
extern char *voirs_get_last_error(void);   /* caller frees with voirs_free_string */
extern void voirs_clear_error(void);
extern const char *voirs_error_message(VoirsErrorCode code); /* static string, do not free */
```

## Building

```bash
# From the workspace root
cargo build --release -p voirs-ffi

# Or from this directory
cargo build --release
```

`[lib] crate-type = ["cdylib", "rlib"]` — the crate produces a dynamic library only (no static
`.a`/`.lib`):

| Platform | Artifact | Import lib |
|---|---|---|
| Linux | `target/release/libvoirs.so` | — |
| macOS | `target/release/libvoirs.dylib` | — |
| Windows | `target/release/voirs.dll` | `target/release/voirs.dll.lib` |

See [`docs/c/linux.md`](docs/c/linux.md), [`docs/c/macos.md`](docs/c/macos.md), and
[`docs/c/windows.md`](docs/c/windows.md) for platform-specific linking, `rpath`/`LD_LIBRARY_PATH`/
`DYLD_LIBRARY_PATH` details, and the full function reference (including the ~9-13 platform-only
functions per OS).

### Python

```bash
pip install maturin
cd crates/voirs-ffi
maturin develop --release --features python   # add ",numpy" for NumPy array support
```

### Node.js

```bash
cd crates/voirs-ffi
cargo build --release --features nodejs
```

## C API overview

The C API is large (roughly 200 functions available on every desktop OS, plus 5-13 more per
platform) because it also exposes internal utilities (zero-copy buffers, format conversion,
allocator control, DSP analysis) as stable C entry points, not just the top-level synthesis path.
Rather than duplicate a 200-row table here, the full categorized listing lives in the platform
guides, which are otherwise identical for the cross-platform functions:

- [`docs/c/linux.md`](docs/c/linux.md)
- [`docs/c/macos.md`](docs/c/macos.md)
- [`docs/c/windows.md`](docs/c/windows.md)

Note there is no plain `voirs_synthesize` function: the one-shot self-contained entry point is
`voirs_synthesize_advanced`, and the callback/streaming entry points are `voirs_synthesize_streaming`,
`voirs_synthesize_streaming_advanced`, and `voirs_synthesize_streaming_realtime` (all in
`src/c_api/synthesis.rs`).

## Memory ownership rules

- Every `voirs_*_create`/`voirs_list_voices`/`voirs_get_voice_info`/`voirs_synthesize_advanced`
  that hands back a heap pointer has a matching `voirs_free_*` — call it exactly once.
- `voirs_free_synthesis_result` also frees `result.audio` and `result.processing_info`; do not
  additionally call `voirs_free_audio_buffer` on `result.audio` or `voirs_free_string` on
  `result.processing_info`.
- Strings returned by `voirs_get_last_error`, `voirs_get_voice`, `voirs_macos_get_system_language`,
  and `voirs_windows_read_registry_config` must be freed with `voirs_free_string`.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE)).
