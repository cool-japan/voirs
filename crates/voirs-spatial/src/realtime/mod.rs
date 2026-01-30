//! Real-time Audio Processing Module
//!
//! This module provides lock-free data structures and zero-copy processing
//! primitives specifically designed for real-time audio applications where
//! latency and deterministic behavior are critical.
//!
//! ## Features
//!
//! - Lock-free SPSC ring buffer for audio streams
//! - Lock-free MPSC ring buffer for multi-source processing
//! - SIMD-aligned memory allocation
//! - Zero-copy audio buffer views
//! - Wait-free audio frame processing
//!
//! ## Performance Targets
//!
//! - VR/AR: <20ms latency, zero allocations in audio thread
//! - Gaming: <30ms latency, minimal allocations
//! - General: <50ms latency, optimized for throughput

pub mod lockfree_buffer;
pub mod simd_allocator;
pub mod zero_copy;

pub use lockfree_buffer::{LockFreeMPSC, LockFreeSPSC, RingBufferError};
pub use simd_allocator::{SimdAlignedBuffer, SimdAllocator, SIMD_ALIGNMENT};
pub use zero_copy::{AudioFrameView, AudioFrameViewMut, ZeroCopyProcessor};
