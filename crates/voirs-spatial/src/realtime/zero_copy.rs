//! Zero-Copy Audio Processing
//!
//! Provides zero-copy views into audio buffers to minimize allocations and
//! memory copies in real-time audio processing pipelines.

use super::simd_allocator::SimdAlignedBuffer;
use std::marker::PhantomData;

/// Immutable zero-copy view into an audio frame
///
/// This allows processing audio data without copying, using only references.
/// Ideal for read-only processing stages like analysis and effects that
/// don't modify the input.
///
/// # Example
///
/// ```rust
/// use voirs_spatial::realtime::AudioFrameView;
///
/// fn analyze_audio(view: AudioFrameView<f32>) -> f32 {
///     // Process without copying
///     view.iter().map(|&x| x * x).sum::<f32>().sqrt()
/// }
/// ```
pub struct AudioFrameView<'a, T> {
    data: &'a [T],
    sample_rate: u32,
    channels: usize,
}

impl<'a, T> AudioFrameView<'a, T> {
    /// Create new audio frame view
    ///
    /// # Arguments
    ///
    /// * `data` - Reference to audio data
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    pub fn new(data: &'a [T], sample_rate: u32, channels: usize) -> Self {
        Self {
            data,
            sample_rate,
            channels,
        }
    }

    /// Get number of samples
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get number of frames (samples per channel)
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.data.len() / self.channels
    }

    /// Get sample rate
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get channel count
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Get raw data slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Get iterator over samples
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Get channel data (interleaved format)
    ///
    /// Returns an iterator over samples for a specific channel
    pub fn channel(&self, ch: usize) -> impl Iterator<Item = &T> {
        assert!(ch < self.channels, "Channel index out of bounds");
        self.data.iter().skip(ch).step_by(self.channels)
    }
}

/// Mutable zero-copy view into an audio frame
///
/// Allows in-place modification of audio data without allocation.
/// Perfect for effects processing that modifies audio in-place.
///
/// # Example
///
/// ```rust
/// use voirs_spatial::realtime::AudioFrameViewMut;
///
/// fn apply_gain(mut view: AudioFrameViewMut<f32>, gain: f32) {
///     for sample in view.iter_mut() {
///         *sample *= gain;
///     }
/// }
/// ```
pub struct AudioFrameViewMut<'a, T> {
    data: &'a mut [T],
    sample_rate: u32,
    channels: usize,
}

impl<'a, T> AudioFrameViewMut<'a, T> {
    /// Create new mutable audio frame view
    pub fn new(data: &'a mut [T], sample_rate: u32, channels: usize) -> Self {
        Self {
            data,
            sample_rate,
            channels,
        }
    }

    /// Get number of samples
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get number of frames
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.data.len() / self.channels
    }

    /// Get sample rate
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get channel count
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Get raw data slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Get mutable raw data slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// Get mutable iterator over samples
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Get mutable channel data
    pub fn channel_mut(&mut self, ch: usize) -> impl Iterator<Item = &mut T> {
        assert!(ch < self.channels, "Channel index out of bounds");
        let channels = self.channels;
        self.data.iter_mut().skip(ch).step_by(channels)
    }

    /// Convert to immutable view
    pub fn as_view(&self) -> AudioFrameView<'_, T> {
        AudioFrameView::new(self.data, self.sample_rate, self.channels)
    }
}

/// Zero-copy audio processor
///
/// Chains multiple processing stages without intermediate allocations.
/// Uses a builder pattern to construct efficient processing pipelines.
///
/// # Example
///
/// ```rust
/// use voirs_spatial::realtime::ZeroCopyProcessor;
///
/// let mut processor: ZeroCopyProcessor<f32> = ZeroCopyProcessor::new(48000, 2);
///
/// processor
///     .add_stage(|view: &mut [f32]| {
///         // Stage 1: Apply gain
///         for sample in view.iter_mut() {
///             *sample *= 0.8;
///         }
///     })
///     .add_stage(|view: &mut [f32]| {
///         // Stage 2: Apply soft clipping
///         for sample in view.iter_mut() {
///             *sample = sample.tanh();
///         }
///     });
/// ```
/// Type alias for processing stage function
type ProcessingStage<T> = Box<dyn FnMut(&mut [T]) + Send>;

/// Zero-copy audio processor for building efficient processing pipelines
pub struct ZeroCopyProcessor<T> {
    /// Processing stages
    stages: Vec<ProcessingStage<T>>,

    /// Sample rate
    sample_rate: u32,

    /// Number of channels
    channels: usize,
}

impl<T: Copy> ZeroCopyProcessor<T> {
    /// Create new zero-copy processor
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        Self {
            stages: Vec::new(),
            sample_rate,
            channels,
        }
    }

    /// Add processing stage
    pub fn add_stage<F>(&mut self, stage: F) -> &mut Self
    where
        F: FnMut(&mut [T]) + Send + 'static,
    {
        self.stages.push(Box::new(stage));
        self
    }

    /// Process audio in-place (zero copy)
    pub fn process(&mut self, data: &mut [T]) {
        for stage in &mut self.stages {
            stage(data);
        }
    }

    /// Process with view
    pub fn process_view(&mut self, view: &mut AudioFrameViewMut<T>) {
        self.process(view.as_mut_slice())
    }

    /// Get number of stages
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Clear all stages
    pub fn clear(&mut self) {
        self.stages.clear();
    }
}

/// Audio frame allocator with zero-copy semantics
///
/// Manages a pool of SIMD-aligned buffers for efficient audio processing
/// with minimal allocation overhead.
pub struct AudioFrameAllocator<T> {
    /// Buffer pool
    pool: Vec<SimdAlignedBuffer<T>>,

    /// Maximum pool size
    max_pool_size: usize,

    /// Buffer size
    buffer_size: usize,
}

impl<T: Copy + Default> AudioFrameAllocator<T> {
    /// Create new allocator
    pub fn new(buffer_size: usize, max_pool_size: usize) -> Self {
        Self {
            pool: Vec::with_capacity(max_pool_size),
            max_pool_size,
            buffer_size,
        }
    }

    /// Allocate or reuse buffer
    pub fn allocate(&mut self) -> SimdAlignedBuffer<T> {
        self.pool
            .pop()
            .unwrap_or_else(|| SimdAlignedBuffer::zeroed(self.buffer_size))
    }

    /// Return buffer to pool
    pub fn deallocate(&mut self, buffer: SimdAlignedBuffer<T>) {
        if self.pool.len() < self.max_pool_size {
            self.pool.push(buffer);
        }
    }

    /// Get pool utilization
    pub fn utilization(&self) -> f32 {
        self.pool.len() as f32 / self.max_pool_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_view() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = AudioFrameView::new(&data, 48000, 2);

        assert_eq!(view.len(), 6);
        assert_eq!(view.num_frames(), 3);
        assert_eq!(view.channels(), 2);
        assert_eq!(view.sample_rate(), 48000);
    }

    #[test]
    fn test_frame_view_mut() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut view = AudioFrameViewMut::new(&mut data, 48000, 2);

        // Modify in place
        for sample in view.iter_mut() {
            *sample *= 2.0;
        }

        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_channel_iteration() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // L R L R L R
        let view = AudioFrameView::new(&data, 48000, 2);

        let left: Vec<f32> = view.channel(0).copied().collect();
        let right: Vec<f32> = view.channel(1).copied().collect();

        assert_eq!(left, vec![1.0, 3.0, 5.0]);
        assert_eq!(right, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_zero_copy_processor() {
        let mut processor = ZeroCopyProcessor::new(48000, 1);

        // Add gain stage
        processor.add_stage(|data| {
            for sample in data.iter_mut() {
                *sample *= 0.5;
            }
        });

        // Add offset stage
        processor.add_stage(|data| {
            for sample in data.iter_mut() {
                *sample += 1.0;
            }
        });

        let mut data = vec![2.0f32, 4.0, 6.0, 8.0];
        processor.process(&mut data);

        // Should apply 0.5 gain then +1.0 offset
        assert_eq!(data, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_frame_allocator() {
        let mut allocator = AudioFrameAllocator::<f32>::new(512, 4);

        let buf1 = allocator.allocate();
        assert_eq!(buf1.len(), 512);

        allocator.deallocate(buf1);
        assert!(allocator.utilization() > 0.0);

        // Should reuse from pool
        let _buf2 = allocator.allocate();
        assert_eq!(allocator.pool.len(), 0); // Pool empty after reuse
    }
}
