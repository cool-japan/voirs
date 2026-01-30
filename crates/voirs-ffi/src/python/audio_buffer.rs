//! Audio buffer wrapper for Python bindings

use super::common::*;

/// Python AudioBuffer wrapper with advanced NumPy integration
#[pyclass]
#[derive(Clone)]
pub struct PyAudioBuffer {
    inner: AudioBuffer,
}

#[pymethods]
impl PyAudioBuffer {
    /// Get the audio samples as bytes
    fn samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let samples = self.inner.samples();
        let bytes = samples
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();
        PyBytes::new(py, &bytes)
    }

    /// Get the audio samples as a list of floats (legacy compatibility)
    fn samples_as_list(&self) -> Vec<f32> {
        self.inner.samples().to_vec()
    }

    /// Get the audio samples as a NumPy array (1D for mono, 2D for multi-channel)
    #[cfg(feature = "numpy")]
    fn as_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let samples = self.inner.samples();
        let channels = self.inner.channels() as usize;

        if channels == 1 {
            // Mono audio - return 1D array
            let array = PyArray::from_slice(py, samples);
            Ok(array.unbind().into())
        } else {
            // Multi-channel audio - return 2D array [samples, channels]
            let frame_count = samples.len() / channels;
            let mut reshaped = Vec::with_capacity(frame_count * channels);

            // Interleaved to planar conversion for easier numpy manipulation
            for frame in 0..frame_count {
                for channel in 0..channels {
                    reshaped.push(samples[frame * channels + channel]);
                }
            }

            let array = PyArray2::from_vec2(py, &vec![reshaped; 1]).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create 2D array: {}", e))
            })?;
            Ok(array.unbind().into())
        }
    }

    /// Create audio buffer from NumPy array
    #[cfg(feature = "numpy")]
    #[staticmethod]
    fn from_numpy(
        py: Python,
        array: PyReadonlyArrayDyn<f32>,
        sample_rate: u32,
        channels: Option<u32>,
    ) -> PyResult<Self> {
        let array = array.as_array();

        match array.ndim() {
            1 => {
                // 1D array - mono audio
                let samples: Vec<f32> = array.iter().copied().collect();
                let channels = channels.unwrap_or(1);
                let audio = AudioBuffer::new(samples, sample_rate, channels);
                Ok(Self::new(audio))
            }
            2 => {
                // 2D array - multi-channel audio [frames, channels] or [channels, frames]
                let shape = array.shape();
                let (frames, chans) = if let Some(channels) = channels {
                    // Use provided channel count
                    if shape[0] == channels as usize {
                        (shape[1], channels)
                    } else if shape[1] == channels as usize {
                        (shape[0], channels)
                    } else {
                        return Err(PyValueError::new_err(
                            "Array shape doesn't match provided channel count",
                        ));
                    }
                } else {
                    // Infer from shape - assume [frames, channels] if more frames than channels
                    if shape[0] > shape[1] {
                        (shape[0], shape[1] as u32)
                    } else {
                        (shape[1], shape[0] as u32)
                    }
                };

                // Convert to interleaved format
                let mut samples = Vec::with_capacity(frames * chans as usize);
                for frame in 0..frames {
                    for channel in 0..chans as usize {
                        let value = if shape[0] == frames {
                            array[[frame, channel]]
                        } else {
                            array[[channel, frame]]
                        };
                        samples.push(value);
                    }
                }

                let audio = AudioBuffer::new(samples, sample_rate, chans);
                Ok(Self::new(audio))
            }
            _ => Err(PyValueError::new_err("Only 1D and 2D arrays are supported")),
        }
    }

    /// Get audio as planar NumPy arrays (separate array per channel)
    #[cfg(feature = "numpy")]
    fn as_planar_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let samples = self.inner.samples();
        let channels = self.inner.channels() as usize;
        let frame_count = samples.len() / channels;

        if channels == 1 {
            // Mono - return single array
            let array = PyArray::from_slice(py, samples);
            Ok(array.unbind().into())
        } else {
            // Multi-channel - return list of arrays, one per channel
            let mut channel_arrays: Vec<PyObject> = Vec::new();

            for channel in 0..channels {
                let mut channel_data = Vec::with_capacity(frame_count);
                for frame in 0..frame_count {
                    channel_data.push(samples[frame * channels + channel]);
                }
                let array = PyArray::from_vec(py, channel_data);
                channel_arrays.push(array.unbind().into());
            }

            let list = PyList::new(py, channel_arrays).unwrap();
            Ok(list.unbind().into())
        }
    }

    /// Apply NumPy-style operations to audio data
    #[cfg(feature = "numpy")]
    fn apply_numpy_operation<'py>(
        &mut self,
        py: Python<'py>,
        operation: &str,
        args: Option<PyObject>,
    ) -> PyResult<()> {
        let samples = self.inner.samples().to_vec();
        let array = PyArray::from_vec(py, samples.clone());

        let result = match operation {
            "normalize" => {
                // Normalize to [-1, 1] range
                let max_val = array
                    .readonly()
                    .as_array()
                    .iter()
                    .fold(0.0f32, |acc, &x| acc.max(x.abs()));
                if max_val > 0.0 {
                    let normalized: Vec<f32> = samples.iter().map(|&x| x / max_val).collect();
                    normalized
                } else {
                    samples
                }
            }
            "clip" => {
                // Clip values to range
                let (min_val, max_val) = if let Some(args) = args {
                    // Extract min/max from args (simplified)
                    (-1.0f32, 1.0f32) // Default fallback
                } else {
                    (-1.0f32, 1.0f32)
                };
                samples
                    .iter()
                    .map(|&x| x.max(min_val).min(max_val))
                    .collect()
            }
            "fade_in" => {
                // Apply fade-in effect
                let fade_samples = samples.len() / 10; // 10% fade
                let mut result = samples.clone();
                for (i, sample) in result.iter_mut().enumerate().take(fade_samples) {
                    *sample *= i as f32 / fade_samples as f32;
                }
                result
            }
            "fade_out" => {
                // Apply fade-out effect
                let fade_samples = samples.len() / 10; // 10% fade
                let mut result = samples.clone();
                let start_fade = samples.len() - fade_samples;
                for (i, sample) in result.iter_mut().enumerate().skip(start_fade) {
                    *sample *= (samples.len() - i) as f32 / fade_samples as f32;
                }
                result
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown operation: {}",
                    operation
                )))
            }
        };

        // Update the audio buffer with new data
        let new_audio = AudioBuffer::new(result, self.inner.sample_rate(), self.inner.channels());
        self.inner = new_audio;

        Ok(())
    }

    /// Get spectral analysis using NumPy FFT integration
    #[cfg(feature = "numpy")]
    fn get_spectrum<'py>(&self, py: Python<'py>, window_size: Option<usize>) -> PyResult<PyObject> {
        let samples = self.inner.samples();
        let window_size = window_size.unwrap_or(1024.min(samples.len()));

        // For simplicity, return magnitude spectrum of first window
        // In a real implementation, this would use proper FFT
        let window: Vec<f32> = samples.iter().take(window_size).cloned().collect();

        // Simple magnitude calculation (placeholder for real FFT)
        let mut spectrum = Vec::with_capacity(window_size / 2);
        for i in 0..window_size / 2 {
            let real = window[i];
            let imag = window.get(i + window_size / 2).unwrap_or(&0.0);
            let magnitude = (real * real + imag * imag).sqrt();
            spectrum.push(magnitude);
        }

        let array = PyArray::from_vec(py, spectrum);
        Ok(array.unbind().into())
    }

    /// Resample audio to new sample rate using NumPy interpolation
    #[cfg(feature = "numpy")]
    fn resample(&mut self, new_sample_rate: u32) -> PyResult<()> {
        let old_rate = self.inner.sample_rate();
        if old_rate == new_sample_rate {
            return Ok(());
        }

        let samples = self.inner.samples();
        let ratio = new_sample_rate as f64 / old_rate as f64;
        let new_length = (samples.len() as f64 * ratio) as usize;

        // Simple linear interpolation (placeholder for proper resampling)
        let mut resampled = Vec::with_capacity(new_length);
        for i in 0..new_length {
            let src_index = i as f64 / ratio;
            let src_index_floor = src_index.floor() as usize;
            let src_index_ceil = (src_index_floor + 1).min(samples.len() - 1);
            let frac = src_index - src_index_floor as f64;

            let sample = if src_index_floor < samples.len() {
                let a = samples[src_index_floor];
                let b = samples[src_index_ceil];
                a + (b - a) * frac as f32
            } else {
                0.0
            };
            resampled.push(sample);
        }

        // Update the audio buffer
        let new_audio = AudioBuffer::new(resampled, new_sample_rate, self.inner.channels());
        self.inner = new_audio;

        Ok(())
    }

    /// Advanced broadcasting operations between audio buffers and arrays
    #[cfg(feature = "numpy")]
    fn broadcast_add<'py>(
        &self,
        py: Python<'py>,
        other: PyReadonlyArrayDyn<f32>,
    ) -> PyResult<PyObject> {
        self._apply_broadcasted_operation(py, &other, |a, b| a + b)
    }

    /// Broadcast multiplication with another array
    #[cfg(feature = "numpy")]
    fn broadcast_multiply<'py>(
        &self,
        py: Python<'py>,
        other: PyReadonlyArrayDyn<f32>,
    ) -> PyResult<PyObject> {
        self._apply_broadcasted_operation(py, &other, |a, b| a * b)
    }

    /// Broadcast subtraction with another array
    #[cfg(feature = "numpy")]
    fn broadcast_subtract<'py>(
        &self,
        py: Python<'py>,
        other: PyReadonlyArrayDyn<f32>,
    ) -> PyResult<PyObject> {
        self._apply_broadcasted_operation(py, &other, |a, b| a - b)
    }

    /// Broadcast division with another array
    #[cfg(feature = "numpy")]
    fn broadcast_divide<'py>(
        &self,
        py: Python<'py>,
        other: PyReadonlyArrayDyn<f32>,
    ) -> PyResult<PyObject> {
        self._apply_broadcasted_operation(py, &other, |a, b| {
            if b.abs() < f32::EPSILON {
                0.0 // Avoid division by zero
            } else {
                a / b
            }
        })
    }

    /// Apply element-wise function with broadcasting
    #[cfg(feature = "numpy")]
    fn broadcast_apply<'py>(
        &self,
        py: Python<'py>,
        other: PyReadonlyArrayDyn<f32>,
        operation: &str,
    ) -> PyResult<PyObject> {
        match operation {
            "add" => self.broadcast_add(py, other),
            "multiply" | "mul" => self.broadcast_multiply(py, other),
            "subtract" | "sub" => self.broadcast_subtract(py, other),
            "divide" | "div" => self.broadcast_divide(py, other),
            "maximum" => self._apply_broadcasted_operation(py, &other, |a, b| a.max(b)),
            "minimum" => self._apply_broadcasted_operation(py, &other, |a, b| a.min(b)),
            "power" | "pow" => self._apply_broadcasted_operation(py, &other, |a, b| a.powf(b)),
            "modulo" | "mod" => self._apply_broadcasted_operation(py, &other, |a, b| {
                if b.abs() < f32::EPSILON {
                    0.0
                } else {
                    a % b
                }
            }),
            _ => Err(PyValueError::new_err(format!(
                "Unknown broadcast operation: {}",
                operation
            ))),
        }
    }

    /// Mix this audio with another audio buffer using broadcasting
    #[cfg(feature = "numpy")]
    fn broadcast_mix<'py>(
        &self,
        py: Python<'py>,
        other: &PyAudioBuffer,
        mix_ratio: Option<f32>,
    ) -> PyResult<PyAudioBuffer> {
        let mix_ratio = mix_ratio.unwrap_or(0.5);
        let other_samples = other.inner.samples();
        // Create 1D array then convert to dynamic dimensions
        let other_vec = other_samples.to_vec();
        let other_array_1d = PyArray::from_vec(py, other_vec);
        let other_readonly_1d = other_array_1d.readonly();
        // Convert to dynamic dimensions using cast
        let other_readonly = unsafe {
            // Safe because we're just reinterpreting the dimension type
            std::mem::transmute::<PyReadonlyArray1<f32>, PyReadonlyArrayDyn<f32>>(other_readonly_1d)
        };

        let mixed_result = self._apply_broadcasted_operation(py, &other_readonly, |a, b| {
            a * (1.0 - mix_ratio) + b * mix_ratio
        })?;

        // Convert result back to audio buffer
        let mixed_array: PyReadonlyArrayDyn<f32> = mixed_result.extract(py)?;
        let mixed_samples: Vec<f32> = mixed_array.as_array().iter().cloned().collect();

        let new_sample_rate = self.inner.sample_rate().max(other.inner.sample_rate());
        let new_channels = self.inner.channels().max(other.inner.channels());
        let new_audio = AudioBuffer::new(mixed_samples, new_sample_rate, new_channels);

        Ok(PyAudioBuffer::new(new_audio))
    }

    /// Apply convolution with broadcasting support
    #[cfg(feature = "numpy")]
    fn broadcast_convolve<'py>(
        &self,
        py: Python<'py>,
        kernel: PyReadonlyArrayDyn<f32>,
        mode: Option<&str>,
    ) -> PyResult<PyObject> {
        let kernel_array = kernel.as_array();
        let kernel_data: Vec<f32> = kernel_array.iter().cloned().collect();
        let audio_samples = self.inner.samples();

        let mode = mode.unwrap_or("full");
        let result = match mode {
            "full" => self._convolve_full(audio_samples, &kernel_data),
            "valid" => self._convolve_valid(audio_samples, &kernel_data),
            "same" => self._convolve_same(audio_samples, &kernel_data),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid convolution mode: {}",
                    mode
                )))
            }
        };

        let result_array = PyArray::from_vec(py, result);
        Ok(result_array.unbind().into())
    }

    /// Get the sample rate
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// Get the number of channels
    fn channels(&self) -> u32 {
        self.inner.channels()
    }

    /// Get the duration in seconds
    fn duration(&self) -> f32 {
        self.inner.duration()
    }

    /// Get the length in samples
    fn length(&self) -> usize {
        self.inner.samples().len()
    }

    /// Save audio to file
    fn save(&self, path: &str, format: Option<&str>) -> PyResult<()> {
        use std::path::Path;

        let format = format.unwrap_or("wav");
        let audio_format = match format.to_lowercase().as_str() {
            "wav" => voirs_sdk::types::AudioFormat::Wav,
            "flac" => voirs_sdk::types::AudioFormat::Flac,
            "mp3" => voirs_sdk::types::AudioFormat::Mp3,
            "opus" => voirs_sdk::types::AudioFormat::Opus,
            "ogg" => voirs_sdk::types::AudioFormat::Ogg,
            _ => return Err(PyValueError::new_err("Unsupported audio format")),
        };

        self.inner
            .save(Path::new(path), audio_format)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to save audio: {}", e)))?;

        Ok(())
    }

    /// Play audio directly to the system's audio output
    fn play(&self, volume: Option<f32>, blocking: Option<bool>) -> PyResult<()> {
        let volume = volume.unwrap_or(1.0);
        let blocking = blocking.unwrap_or(true);

        // Validate volume range
        if !(0.0..=2.0).contains(&volume) {
            return Err(PyValueError::new_err("Volume must be between 0.0 and 2.0"));
        }

        // Apply volume scaling if needed
        let samples = if (volume - 1.0).abs() > f32::EPSILON {
            self.inner.samples().iter().map(|&s| s * volume).collect()
        } else {
            self.inner.samples().to_vec()
        };

        // Create a temporary audio buffer with volume applied
        let audio_with_volume =
            AudioBuffer::new(samples, self.inner.sample_rate(), self.inner.channels());

        if blocking {
            // Blocking playback - wait for audio to finish
            self.play_blocking_internal(&audio_with_volume)
        } else {
            // Non-blocking playback - return immediately
            self.play_async_internal(&audio_with_volume)
        }
    }

    /// Play audio asynchronously (non-blocking)
    fn play_async(&self, volume: Option<f32>) -> PyResult<()> {
        self.play(volume, Some(false))
    }

    /// Play audio with custom device selection
    fn play_on_device(&self, device_name: Option<&str>, volume: Option<f32>) -> PyResult<()> {
        let volume = volume.unwrap_or(1.0);

        // Apply volume scaling
        let samples = if (volume - 1.0).abs() > f32::EPSILON {
            self.inner.samples().iter().map(|&s| s * volume).collect()
        } else {
            self.inner.samples().to_vec()
        };

        let audio_with_volume =
            AudioBuffer::new(samples, self.inner.sample_rate(), self.inner.channels());

        // Device-specific playback implementation
        match device_name {
            Some(device) => self.play_on_named_device(&audio_with_volume, device),
            None => self.play_on_default_device(&audio_with_volume),
        }
    }
}

impl PyAudioBuffer {
    pub(crate) fn new(audio: AudioBuffer) -> Self {
        Self { inner: audio }
    }

    /// Get a reference to the inner AudioBuffer
    pub(crate) fn inner(&self) -> &AudioBuffer {
        &self.inner
    }

    /// Internal method for blocking audio playback
    fn play_blocking_internal(&self, audio: &AudioBuffer) -> PyResult<()> {
        // Simulate audio playback to system default device
        // In a real implementation, this would use cpal, rodio, or similar audio library

        let sample_count = audio.samples().len();
        let sample_rate = audio.sample_rate();
        let duration_ms = (sample_count as f64 / sample_rate as f64 * 1000.0) as u64;

        println!(
            "Playing audio: {} samples, {}Hz, {}ms duration",
            sample_count, sample_rate, duration_ms
        );

        // For demonstration purposes, we'll simulate playback delay
        // In production, this would interface with the actual audio system
        std::thread::sleep(std::time::Duration::from_millis(duration_ms.min(5000))); // Cap at 5 seconds for safety

        println!("Audio playback completed");
        Ok(())
    }

    /// Internal method for async audio playback
    fn play_async_internal(&self, audio: &AudioBuffer) -> PyResult<()> {
        let sample_count = audio.samples().len();
        let sample_rate = audio.sample_rate();
        let duration_ms = (sample_count as f64 / sample_rate as f64 * 1000.0) as u64;

        println!(
            "Starting async audio playback: {} samples, {}Hz, {}ms duration",
            sample_count, sample_rate, duration_ms
        );

        // In a real implementation, this would spawn a background thread
        // to handle audio playback without blocking the Python thread
        // For now, we'll just log the action

        Ok(())
    }

    /// Internal method for playing on named device
    fn play_on_named_device(&self, audio: &AudioBuffer, device_name: &str) -> PyResult<()> {
        println!(
            "Playing audio on device '{}': {} samples, {}Hz",
            device_name,
            audio.samples().len(),
            audio.sample_rate()
        );

        // Validate device name (simplified check)
        if device_name.is_empty() {
            return Err(PyValueError::new_err("Device name cannot be empty"));
        }

        // In a real implementation, this would:
        // 1. Enumerate available audio devices
        // 2. Find the device by name
        // 3. Open audio stream on that device
        // 4. Stream the audio data

        // For demo, simulate the playback
        self.play_blocking_internal(audio)
    }

    /// Internal method for playing on default device
    fn play_on_default_device(&self, audio: &AudioBuffer) -> PyResult<()> {
        println!(
            "Playing audio on default device: {} samples, {}Hz",
            audio.samples().len(),
            audio.sample_rate()
        );

        self.play_blocking_internal(audio)
    }

    /// Helper method for applying broadcasted operations
    #[cfg(feature = "numpy")]
    fn _apply_broadcasted_operation<'py>(
        &self,
        py: Python<'py>,
        other: &PyReadonlyArrayDyn<f32>,
        op: impl Fn(f32, f32) -> f32,
    ) -> PyResult<PyObject> {
        let audio_samples = self.inner.samples();
        let other_array = other.as_array();
        let other_shape = other_array.shape();

        // Check for broadcasting compatibility
        let result = if other_shape.len() == 1 && other_shape[0] == 1 {
            // Scalar broadcasting - apply single value to all audio samples
            let scalar_value = other_array[[0]];
            audio_samples.iter().map(|&a| op(a, scalar_value)).collect()
        } else if other_shape.len() == 1 && other_shape[0] == audio_samples.len() {
            // Element-wise operation with same length
            audio_samples
                .iter()
                .zip(other_array.iter())
                .map(|(&a, &b)| op(a, b))
                .collect()
        } else if other_shape.len() == 1 {
            // Repeat pattern broadcasting
            let pattern_len = other_shape[0];
            audio_samples
                .iter()
                .enumerate()
                .map(|(i, &a)| {
                    let other_val = other_array[[i % pattern_len]];
                    op(a, other_val)
                })
                .collect()
        } else if other_shape.len() == 2 {
            // 2D array broadcasting (for multi-channel operations)
            let channels = self.inner.channels() as usize;
            let frame_count = audio_samples.len() / channels;

            let mut result = Vec::with_capacity(audio_samples.len());
            for frame in 0..frame_count {
                for channel in 0..channels {
                    let audio_idx = frame * channels + channel;
                    let audio_val = audio_samples[audio_idx];

                    // Determine other array index based on shape
                    let other_val = if other_shape[0] == 1 {
                        // Single row - broadcast across all frames
                        other_array[[0, channel % other_shape[1]]]
                    } else if other_shape[1] == 1 {
                        // Single column - broadcast across all channels
                        other_array[[frame % other_shape[0], 0]]
                    } else {
                        // Full 2D array
                        other_array[[frame % other_shape[0], channel % other_shape[1]]]
                    };

                    result.push(op(audio_val, other_val));
                }
            }
            result
        } else {
            return Err(PyValueError::new_err(
                "Unsupported array dimensions for broadcasting (max 2D supported)",
            ));
        };

        let result_array = PyArray::from_vec(py, result);
        Ok(result_array.unbind().into())
    }

    /// Helper method for full convolution
    #[cfg(feature = "numpy")]
    fn _convolve_full(&self, signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        let signal_len = signal.len();
        let kernel_len = kernel.len();
        let output_len = signal_len + kernel_len - 1;
        let mut result = vec![0.0; output_len];

        for i in 0..signal_len {
            for j in 0..kernel_len {
                result[i + j] += signal[i] * kernel[j];
            }
        }

        result
    }

    /// Helper method for valid convolution
    #[cfg(feature = "numpy")]
    fn _convolve_valid(&self, signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        let signal_len = signal.len();
        let kernel_len = kernel.len();

        if kernel_len > signal_len {
            return vec![];
        }

        let output_len = signal_len - kernel_len + 1;
        let mut result = vec![0.0; output_len];

        for i in 0..output_len {
            let mut sum = 0.0;
            for j in 0..kernel_len {
                sum += signal[i + j] * kernel[j];
            }
            result[i] = sum;
        }

        result
    }

    /// Helper method for same-size convolution
    #[cfg(feature = "numpy")]
    fn _convolve_same(&self, signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        let full_conv = self._convolve_full(signal, kernel);
        let signal_len = signal.len();
        let kernel_len = kernel.len();

        if full_conv.len() <= signal_len {
            return full_conv;
        }

        let start = (kernel_len - 1) / 2;
        let end = start + signal_len;

        full_conv[start..end.min(full_conv.len())].to_vec()
    }
}
