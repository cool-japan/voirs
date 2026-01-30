//! Streaming processor for Python bindings

use super::common::*;

/// Advanced NumPy-based streaming audio processor
#[cfg(feature = "numpy")]
#[pyclass]
pub struct PyStreamingProcessor {
    chunk_size: usize,
    sample_rate: u32,
    channels: u32,
    buffer: Vec<f32>,
    callback: Option<PyObject>,
}

#[cfg(feature = "numpy")]
#[pymethods]
impl PyStreamingProcessor {
    #[new]
    fn new(chunk_size: usize, sample_rate: u32, channels: u32) -> Self {
        Self {
            chunk_size,
            sample_rate,
            channels,
            buffer: Vec::new(),
            callback: None,
        }
    }

    /// Set a Python callback for processing audio chunks
    fn set_callback(&mut self, callback: PyObject) {
        self.callback = Some(callback);
    }

    /// Process an audio chunk with NumPy array
    fn process_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk: PyReadonlyArray1<f32>,
    ) -> PyResult<PyObject> {
        let chunk_data = chunk.as_array().to_vec();

        if let Some(ref callback) = self.callback {
            // Convert to NumPy array and call Python callback
            let input_array = PyArray::from_vec(py, chunk_data.clone());
            let result = callback.call1(py, (input_array.clone(),))?;

            // Extract processed data
            if let Ok(processed) = result.extract::<PyReadonlyArray1<f32>>(py) {
                let processed_data = processed.as_array().to_vec();
                let output_array = PyArray::from_vec(py, processed_data);
                Ok(output_array.unbind().into())
            } else {
                // Return original data if callback didn't return array
                Ok(input_array.unbind().into())
            }
        } else {
            // No callback - return original chunk
            let array = PyArray::from_vec(py, chunk_data);
            Ok(array.unbind().into())
        }
    }

    /// Add data to internal buffer and process when chunk is ready
    fn add_samples<'py>(
        &mut self,
        py: Python<'py>,
        samples: PyReadonlyArray1<f32>,
    ) -> PyResult<Option<PyObject>> {
        let new_samples = samples.as_array().to_vec();
        self.buffer.extend(new_samples);

        if self.buffer.len() >= self.chunk_size {
            // Extract chunk and process it
            let chunk: Vec<f32> = self.buffer.drain(0..self.chunk_size).collect();
            let chunk_array = PyArray::from_vec(py, chunk);
            let processed = self.process_chunk(py, chunk_array.readonly())?;
            Ok(Some(processed))
        } else {
            Ok(None)
        }
    }

    /// Get remaining buffered samples
    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        if !self.buffer.is_empty() {
            let remaining = self.buffer.clone();
            self.buffer.clear();
            let array = PyArray::from_vec(py, remaining);
            Ok(Some(array.unbind().into()))
        } else {
            Ok(None)
        }
    }
}
