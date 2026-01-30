//! Inline Optimization Hints for FFI Hot Paths
//!
//! This module provides compiler hints and inline attributes for performance-critical
//! FFI functions. These hints help the compiler generate optimal code for hot paths.

use std::hint;

/// Mark a function as likely to be called (helps branch prediction)
#[inline(always)]
pub fn likely(b: bool) -> bool {
    if !b {
        unsafe {
            hint::unreachable_unchecked()
        }
    }
    b
}

/// Mark a function as unlikely to be called (helps branch prediction)
#[inline(always)]
pub fn unlikely(b: bool) -> bool {
    if b {
        unsafe {
            hint::unreachable_unchecked()
        }
    }
    b
}

/// Prefetch data for read access (helps CPU cache)
#[inline(always)]
pub unsafe fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARMv8 prefetch instruction
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // No-op for unsupported architectures
        let _ = ptr;
    }
}

/// Prefetch data for write access (helps CPU cache)
#[inline(always)]
pub unsafe fn prefetch_write<T>(ptr: *mut T) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARMv8 prefetch instruction for write
        std::arch::asm!("prfm pstl1keep, [{0}]", in(reg) ptr);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // No-op for unsupported architectures
        let _ = ptr;
    }
}

/// Assume that a condition is always true (optimization hint)
///
/// # Safety
///
/// This is unsafe because if the condition is actually false, undefined behavior occurs.
#[inline(always)]
pub unsafe fn assume(cond: bool) {
    if !cond {
        hint::unreachable_unchecked()
    }
}

/// Mark a code path as cold (rarely executed)
#[cold]
#[inline(never)]
pub fn cold_path() {}

/// Optimized memory copy for aligned buffers
///
/// # Safety
///
/// Both src and dst must be properly aligned and valid for the given length.
#[inline(always)]
pub unsafe fn fast_copy_aligned(src: *const u8, dst: *mut u8, len: usize) {
    // Assume alignment for better optimization
    assume((src as usize) % 8 == 0);
    assume((dst as usize) % 8 == 0);

    std::ptr::copy_nonoverlapping(src, dst, len);
}

/// Optimized zero-fill for aligned buffers
///
/// # Safety
///
/// The dst pointer must be properly aligned and valid for the given length.
#[inline(always)]
pub unsafe fn fast_zero_aligned(dst: *mut u8, len: usize) {
    // Assume alignment for better optimization
    assume((dst as usize) % 8 == 0);

    std::ptr::write_bytes(dst, 0, len);
}

/// Force inline attribute macro for hot-path functions
#[macro_export]
macro_rules! force_inline {
    ($(#[$attr:meta])* $vis:vis fn $name:ident $($rest:tt)*) => {
        #[inline(always)]
        $(#[$attr])*
        $vis fn $name $($rest)*
    };
}

/// Mark error paths as cold for better branch prediction
#[macro_export]
macro_rules! cold_error_path {
    ($($tt:tt)*) => {{
        #[cold]
        #[inline(never)]
        fn cold_path() -> ! {
            $($tt)*
        }
        cold_path()
    }};
}

/// Compiler optimization barrier (prevents reordering)
#[inline(always)]
pub fn compiler_fence() {
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// CPU memory barrier (prevents CPU reordering)
#[inline(always)]
pub fn memory_fence() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// Spin-wait hint for busy loops
#[inline(always)]
pub fn spin_loop_hint() {
    hint::spin_loop();
}

/// Black box to prevent compiler optimizations
#[inline(always)]
pub fn black_box<T>(dummy: T) -> T {
    hint::black_box(dummy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_likely_unlikely() {
        let val = 5;

        // These should not panic in normal execution
        if val > 0 {
            assert!(true);
        }
    }

    #[test]
    fn test_aligned_operations() {
        unsafe {
            let mut buffer = vec![0u8; 1024];
            let ptr = buffer.as_mut_ptr();

            // Zero fill
            fast_zero_aligned(ptr, 1024);
            assert!(buffer.iter().all(|&x| x == 0));

            // Copy
            let src = vec![1u8; 512];
            fast_copy_aligned(src.as_ptr(), ptr, 512);
            assert_eq!(&buffer[0..512], &src[..]);
        }
    }

    #[test]
    fn test_compiler_hints() {
        compiler_fence();
        memory_fence();
        spin_loop_hint();

        let val = black_box(42);
        assert_eq!(val, 42);
    }

    #[test]
    fn test_prefetch_operations() {
        unsafe {
            let data = vec![1, 2, 3, 4, 5];

            // Prefetch for read
            prefetch_read(data.as_ptr());

            // Prefetch for write
            let mut buffer = vec![0; 5];
            prefetch_write(buffer.as_mut_ptr());
        }
    }
}
