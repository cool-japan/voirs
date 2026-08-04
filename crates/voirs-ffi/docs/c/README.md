# VoiRS C API Documentation

This directory documents the real, currently-exported C API of the `voirs-ffi` crate (library
name `voirs`). It replaces an earlier revision that linked to files which never existed
(`api_reference.md`, `data_types.md`, `quick_start.md`, `error_handling.md`,
`memory_management.md`, `threading.md`, `performance.md`, an `examples/` directory) and that
called functions with no real implementation (`voirs_create_pipeline(NULL)` taking an argument it
doesn't take, `voirs_synthesize(pipeline, text)`, `voirs_save_audio`, `voirs_destroy_audio_buffer`,
`voirs_synthesize_to_file`, `voirs_get_last_error_message`, `voirs_get_performance_metrics`, and
more). Every function named below was verified against `crates/voirs-ffi/src/**/*.rs` directly.

No C header ships with this crate (there is no `cbindgen` build step), so callers declare
prototypes themselves — see the [crate README's Quick Start](../../README.md#quick-start) for a
complete, compilable example that does exactly that.

## Documentation in this directory

- [`../../README.md`](../../README.md) — crate overview, feature flags, Quick Start (C and
  Python), pipeline-handle vs. self-contained synthesis, error handling, memory ownership rules.
- [**Linux Integration**](linux.md) — building, linking (`LD_LIBRARY_PATH`/rpath), the 9
  `voirs_linux_*` functions, and the full cross-platform function index.
- [**macOS Integration**](macos.md) — building, linking (`DYLD_LIBRARY_PATH`/rpath/codesigning),
  the 9 `voirs_macos_*` functions, and the full cross-platform function index.
- [**Windows Integration**](windows.md) — building (including the `windows-platform` feature
  requirement), linking (MSVC/MinGW), the 5 `voirs_windows_*` functions, and the full
  cross-platform function index.

## Getting started

```bash
cargo build --release -p voirs-ffi
```

Produces `target/release/libvoirs.so` (Linux), `libvoirs.dylib` (macOS), or `voirs.dll` +
`voirs.dll.lib` (Windows) — `[lib] name = "voirs"`, `crate-type = ["cdylib", "rlib"]`.

```bash
gcc example.c -Ltarget/release -lvoirs -o example      # Linux/macOS
cl example.c /link /LIBPATH:target\release voirs.dll.lib  # Windows/MSVC
```

See the platform pages above for runtime library-path setup (`LD_LIBRARY_PATH`,
`DYLD_LIBRARY_PATH`, or DLL placement/`PATH`).

## Memory management (`c_api::allocator` and `c_api::memory`)

These functions are real and always compiled (no feature flag needed):

```c
/* Choose a pluggable global allocator strategy. Constants from
 * crates/voirs-ffi/src/c_api/allocator.rs. */
#define VOIRS_ALLOCATOR_SYSTEM          0
#define VOIRS_ALLOCATOR_POOL            1
#define VOIRS_ALLOCATOR_DEBUG           2
#define VOIRS_ALLOCATOR_TRACKED_SYSTEM  3

typedef struct {
    unsigned int total_allocations;
    unsigned int total_deallocations;
    unsigned int current_allocations;
    unsigned int peak_allocations;
    unsigned int total_bytes_allocated;
    unsigned int total_bytes_deallocated;
    unsigned int current_bytes_allocated;
    unsigned int peak_bytes_allocated;
} VoirsAllocatorStats;

extern int voirs_set_allocator(int allocator_type, unsigned int block_size,
                                unsigned int blocks_per_chunk, int enable_backtrace);
extern int voirs_get_allocator_stats(VoirsAllocatorStats *stats);
extern int voirs_reset_allocator_stats(void);
extern char *voirs_memory_get_stats(void); /* JSON string; free with voirs_free_string */
extern void voirs_free_string(char *s);
```

```c
voirs_set_allocator(VOIRS_ALLOCATOR_POOL, /*block_size=*/4096, /*blocks_per_chunk=*/50, 0);

VoirsAllocatorStats stats;
voirs_get_allocator_stats(&stats);
printf("current bytes allocated: %u\n", stats.current_bytes_allocated);

voirs_reset_allocator_stats();

char *stats_json = voirs_memory_get_stats();
printf("%s\n", stats_json);
voirs_free_string(stats_json);
```

Note: an earlier revision of this document described a `perf`-module family of APIs (lock-free
audio rings, an adaptive allocator, a memory profiler, a generic
`voirs_create_memory_pool`/`voirs_pool_allocate`). That module was never wired into the compiled
crate (`mod perf;` was never declared in `lib.rs`), so none of those symbols were ever actually
linkable, and it has since been removed from the source tree entirely rather than left to bit-rot
unreachable. The allocator/memory APIs shown above are the real, shipped equivalent for allocator
selection and allocation statistics.

## Error handling

```c
extern int voirs_has_error(void);
extern char *voirs_get_last_error(void);   /* caller frees with voirs_free_string */
extern void voirs_clear_error(void);
```

Error codes (`VoirsErrorCode`, `crates/voirs-ffi/src/lib.rs`) are `#[repr(C)]` with explicit
values: `VOIRS_SUCCESS = 0`, `VOIRS_ERROR_INVALID_PARAMETER = 1`,
`VOIRS_ERROR_INITIALIZATION_FAILED = 2`, `VOIRS_ERROR_SYNTHESIS_FAILED = 3`,
`VOIRS_ERROR_VOICE_NOT_FOUND = 4`, `VOIRS_ERROR_IO_ERROR = 5`, `VOIRS_ERROR_OUT_OF_MEMORY = 6`,
`VOIRS_ERROR_OPERATION_CANCELLED = 7`, `VOIRS_ERROR_INTERNAL_ERROR = 99`.

## Full function reference

The C API is large — roughly 199 functions available on every desktop OS, plus 5-13 more per
platform. Rather than repeat the same ~199-entry categorized index on every page, it lives once,
identically, at the bottom of [`linux.md`](linux.md), [`macos.md`](macos.md), and
[`windows.md`](windows.md), alongside each platform's own extra functions.

## Support

- Source repository: <https://github.com/cool-japan/voirs>
