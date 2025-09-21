//! Buffer management for real-time audio processing

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// Buffer configuration for real-time processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Buffer size in samples
    pub buffer_size: usize,
    /// Number of buffers in the ring buffer
    pub num_buffers: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// Buffer overlap in samples
    pub overlap: usize,
    /// Windowing function
    pub window_function: WindowFunction,
    /// Buffer management strategy
    pub buffer_strategy: BufferStrategy,
}

/// Windowing functions for buffer processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowFunction {
    /// Rectangular window
    Rectangular,
    /// Hanning window
    Hanning,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Kaiser window
    Kaiser { beta: f32 },
    /// Gaussian window
    Gaussian { sigma: f32 },
}

/// Buffer management strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferStrategy {
    /// Circular buffer
    Circular,
    /// Double buffering
    DoubleBuffering,
    /// Triple buffering
    TripleBuffering,
    /// Adaptive buffering
    Adaptive,
}

/// Real-time audio buffer
#[derive(Debug, Clone)]
pub struct RealTimeBuffer {
    /// Buffer data
    pub data: VecDeque<f32>,
    /// Buffer capacity
    pub capacity: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// Current position
    pub position: usize,
    /// Timestamp
    pub timestamp: Instant,
}

impl RealTimeBuffer {
    /// Create a new real-time buffer
    pub fn new(capacity: usize, sample_rate: u32, channels: u32) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
            sample_rate,
            channels,
            position: 0,
            timestamp: Instant::now(),
        }
    }

    /// Add samples to the buffer
    pub fn push_samples(&mut self, samples: &[f32]) -> crate::Result<()> {
        for &sample in samples {
            if self.data.len() >= self.capacity {
                self.data.pop_front();
            }
            self.data.push_back(sample);
        }
        self.timestamp = Instant::now();
        Ok(())
    }

    /// Get the current buffer size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get buffer utilization as a percentage
    pub fn utilization(&self) -> f32 {
        self.data.len() as f32 / self.capacity as f32 * 100.0
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.data.clear();
        self.position = 0;
        self.timestamp = Instant::now();
    }
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            num_buffers: 3,
            sample_rate: 44100,
            channels: 2,
            overlap: 256,
            window_function: WindowFunction::Hanning,
            buffer_strategy: BufferStrategy::DoubleBuffering,
        }
    }
}
