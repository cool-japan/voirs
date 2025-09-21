#ifndef VOIRS_FFI_PERF_H
#define VOIRS_FFI_PERF_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct BatchSynthesisResult BatchSynthesisResult;
typedef struct FfiStats FfiStats;
typedef struct SimdAudioProcessor SimdAudioProcessor;
typedef struct LruCache LruCache;

// Batch processing functions
BatchSynthesisResult* voirs_ffi_process_batch_synthesis(
    const char* const* texts,
    size_t count,
    const char* voice_id
);

void voirs_ffi_destroy_batch_result(BatchSynthesisResult* result);

// Statistics functions
FfiStats* voirs_ffi_get_stats(void);
void voirs_ffi_destroy_stats(FfiStats* stats);

// SIMD audio processing functions
SimdAudioProcessor* voirs_ffi_create_simd_processor(void);
void voirs_ffi_destroy_simd_processor(SimdAudioProcessor* processor);

bool voirs_ffi_simd_apply_gain(
    SimdAudioProcessor* processor,
    float* samples,
    size_t count,
    float gain
);

bool voirs_ffi_simd_mix_buffers(
    SimdAudioProcessor* processor,
    float* dest,
    const float* src,
    size_t count,
    float mix_ratio
);

// Cache optimization functions
LruCache* voirs_ffi_create_lru_cache(size_t capacity);
void voirs_ffi_destroy_lru_cache(LruCache* cache);

#ifdef __cplusplus
}
#endif

#endif // VOIRS_FFI_PERF_H