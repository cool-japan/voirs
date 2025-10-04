use crate::wasm::recognizer::{WasmRecognitionResult, WasmStreamingConfig};
use crate::wasm::utils::{console_error, console_log};
use js_sys::{Array, Uint8Array};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use wasm_bindgen::prelude::*;
use web_sys::{console, AudioContext, AudioWorkletNode};

#[derive(Serialize, Deserialize)]
/// Streaming State
pub struct StreamingState {
    /// is active
    pub is_active: bool,
    /// chunk count
    pub chunk_count: usize,
    /// total audio duration
    pub total_audio_duration: f64,
    /// buffer size
    pub buffer_size: usize,
    /// processing latency
    pub processing_latency: f64,
}

#[derive(Serialize, Deserialize)]
/// Streaming Metrics
pub struct StreamingMetrics {
    /// chunks processed
    pub chunks_processed: usize,
    /// average latency
    pub average_latency: f64,
    /// peak latency
    pub peak_latency: f64,
    /// buffer underruns
    pub buffer_underruns: usize,
    /// processing errors
    pub processing_errors: usize,
    /// total audio time
    pub total_audio_time: f64,
    /// real time factor
    pub real_time_factor: f64,
}

#[wasm_bindgen]
/// Wasm Streaming Processor
pub struct WasmStreamingProcessor {
    audio_context: Option<AudioContext>,
    audio_buffer: VecDeque<f32>,
    config: WasmStreamingConfig,
    state: StreamingState,
    metrics: StreamingMetrics,
    chunk_size: usize,
    overlap_size: usize,
    sample_rate: f32,
}

#[wasm_bindgen]
impl WasmStreamingProcessor {
    #[wasm_bindgen(constructor)]
    /// new
    pub fn new(config: JsValue) -> Result<WasmStreamingProcessor, JsValue> {
        let config: WasmStreamingConfig = config
            .into_serde()
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {e}")))?;

        let sample_rate = 16000.0; // Default sample rate
        let chunk_duration = config.chunk_duration.unwrap_or(1.0); // 1 second default
        let overlap_duration = config.overlap_duration.unwrap_or(0.1); // 100ms default

        let chunk_size = (chunk_duration * sample_rate) as usize;
        let overlap_size = (overlap_duration * sample_rate) as usize;

        Ok(WasmStreamingProcessor {
            audio_context: None,
            audio_buffer: VecDeque::new(),
            config,
            state: StreamingState {
                is_active: false,
                chunk_count: 0,
                total_audio_duration: 0.0,
                buffer_size: 0,
                processing_latency: 0.0,
            },
            metrics: StreamingMetrics {
                chunks_processed: 0,
                average_latency: 0.0,
                peak_latency: 0.0,
                buffer_underruns: 0,
                processing_errors: 0,
                total_audio_time: 0.0,
                real_time_factor: 0.0,
            },
            chunk_size,
            overlap_size,
            sample_rate,
        })
    }

    #[wasm_bindgen]
    /// initialize audio context
    pub async fn initialize_audio_context(&mut self) -> Result<(), JsValue> {
        match AudioContext::new() {
            Ok(ctx) => {
                self.audio_context = Some(ctx);
                console_log!("Streaming audio context initialized");
                Ok(())
            }
            Err(e) => {
                console_error!("Failed to create streaming audio context: {:?}", e);
                Err(JsValue::from_str("Failed to create audio context"))
            }
        }
    }

    #[wasm_bindgen]
    /// start streaming
    pub fn start_streaming(&mut self) {
        self.state.is_active = true;
        self.state.chunk_count = 0;
        self.audio_buffer.clear();
        console_log!("Streaming processor started");
    }

    #[wasm_bindgen]
    /// stop streaming
    pub fn stop_streaming(&mut self) {
        self.state.is_active = false;
        console_log!("Streaming processor stopped");
    }

    #[wasm_bindgen]
    /// add audio data
    pub fn add_audio_data(&mut self, audio_data: &[f32]) -> bool {
        if !self.state.is_active {
            return false;
        }

        // Add audio data to buffer
        for &sample in audio_data {
            self.audio_buffer.push_back(sample);
        }

        self.state.buffer_size = self.audio_buffer.len();

        // Update audio duration
        let new_duration = audio_data.len() as f64 / self.sample_rate as f64;
        self.state.total_audio_duration += new_duration;
        self.metrics.total_audio_time += new_duration;

        true
    }

    #[wasm_bindgen]
    /// get next chunk
    pub fn get_next_chunk(&mut self) -> Option<Array> {
        if !self.state.is_active {
            return None;
        }

        // Check if we have enough data for a chunk
        if self.audio_buffer.len() < self.chunk_size {
            return None;
        }

        // Extract chunk with overlap
        let mut chunk = Vec::with_capacity(self.chunk_size);

        // For overlapping chunks, keep some data from previous chunk
        let start_index = if self.state.chunk_count > 0 {
            self.chunk_size - self.overlap_size
        } else {
            0
        };

        // Extract the chunk
        for _ in 0..self.chunk_size {
            if let Some(sample) = self.audio_buffer.pop_front() {
                chunk.push(sample);
            } else {
                break;
            }
        }

        if chunk.len() == self.chunk_size {
            self.state.chunk_count += 1;
            self.state.buffer_size = self.audio_buffer.len();

            // Convert to JavaScript Array
            let js_array = Array::new();
            for sample in chunk {
                js_array.push(&JsValue::from_f64(sample as f64));
            }

            Some(js_array)
        } else {
            // Put back the incomplete chunk
            for sample in chunk.into_iter().rev() {
                self.audio_buffer.push_front(sample);
            }
            None
        }
    }

    #[wasm_bindgen]
    /// update processing metrics
    pub fn update_processing_metrics(&mut self, latency_ms: f64) {
        self.state.processing_latency = latency_ms;
        self.metrics.chunks_processed += 1;

        // Update average latency
        let total_latency =
            self.metrics.average_latency * (self.metrics.chunks_processed - 1) as f64 + latency_ms;
        self.metrics.average_latency = total_latency / self.metrics.chunks_processed as f64;

        // Update peak latency
        if latency_ms > self.metrics.peak_latency {
            self.metrics.peak_latency = latency_ms;
        }

        // Calculate real-time factor
        let chunk_duration_ms = (self.chunk_size as f64 / self.sample_rate as f64) * 1000.0;
        self.metrics.real_time_factor = latency_ms / chunk_duration_ms;
    }

    #[wasm_bindgen]
    /// report error
    pub fn report_error(&mut self) {
        self.metrics.processing_errors += 1;
    }

    #[wasm_bindgen]
    /// report buffer underrun
    pub fn report_buffer_underrun(&mut self) {
        self.metrics.buffer_underruns += 1;
    }

    #[wasm_bindgen]
    /// get state
    pub fn get_state(&self) -> JsValue {
        JsValue::from_serde(&self.state).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    /// get metrics
    pub fn get_metrics(&self) -> JsValue {
        JsValue::from_serde(&self.metrics).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    /// get buffer info
    pub fn get_buffer_info(&self) -> JsValue {
        let info = serde_json::json!({
            "buffer_size": self.audio_buffer.len(),
            "chunk_size": self.chunk_size,
            "overlap_size": self.overlap_size,
            "sample_rate": self.sample_rate,
            "chunks_available": self.audio_buffer.len() / self.chunk_size,
            "buffer_duration_ms": (self.audio_buffer.len() as f64 / self.sample_rate as f64) * 1000.0
        });

        JsValue::from_serde(&info).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    /// reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = StreamingMetrics {
            chunks_processed: 0,
            average_latency: 0.0,
            peak_latency: 0.0,
            buffer_underruns: 0,
            processing_errors: 0,
            total_audio_time: 0.0,
            real_time_factor: 0.0,
        };
    }

    #[wasm_bindgen]
    /// configure adaptive quality
    pub fn configure_adaptive_quality(&mut self, enable: bool) {
        if let Some(ref mut config) = self.config.quality_adaptive {
            *config = enable;
        } else {
            self.config.quality_adaptive = Some(enable);
        }

        console_log!(
            "Adaptive quality {}",
            if enable { "enabled" } else { "disabled" }
        );
    }

    #[wasm_bindgen]
    /// adjust chunk size
    pub fn adjust_chunk_size(&mut self, new_duration_ms: f32) {
        let new_chunk_size = ((new_duration_ms / 1000.0) * self.sample_rate) as usize;

        if new_chunk_size > 0 && new_chunk_size != self.chunk_size {
            self.chunk_size = new_chunk_size;
            console_log!(
                "Chunk size adjusted to {} samples ({:.1}ms)",
                self.chunk_size,
                new_duration_ms
            );
        }
    }

    #[wasm_bindgen]
    /// is ready for processing
    pub fn is_ready_for_processing(&self) -> bool {
        self.state.is_active && self.audio_buffer.len() >= self.chunk_size
    }

    #[wasm_bindgen]
    /// get processing recommendations
    pub fn get_processing_recommendations(&self) -> JsValue {
        let mut recommendations = Vec::new();

        if self.metrics.real_time_factor > 1.0 {
            recommendations.push("Consider using a smaller model or reducing chunk size");
        }

        if self.metrics.buffer_underruns > 0 {
            recommendations.push("Increase buffer size to prevent audio dropouts");
        }

        if self.metrics.average_latency > 500.0 {
            recommendations.push("High processing latency detected, consider optimization");
        }

        if self.audio_buffer.len() > self.chunk_size * 10 {
            recommendations.push("Large buffer accumulation, check processing speed");
        }

        JsValue::from_serde(&recommendations).unwrap_or(JsValue::NULL)
    }
}

// Helper functions for streaming audio processing

#[wasm_bindgen]
/// create audio worklet processor
pub fn create_audio_worklet_processor() -> String {
    r#"
class VoirsAudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.chunkSize = options.processorOptions?.chunkSize || 1024;
        this.buffer = new Float32Array(this.chunkSize * 4); // 4x chunk size buffer
        this.bufferIndex = 0;
        this.initialized = false;
        
        this.port.onmessage = (event) => {
            if (event.data.type === 'init') {
                this.initialized = true;
                console.log('VoiRS Audio Processor initialized');
            }
        };
    }
    
    process(inputs, outputs, parameters) {
        if (!this.initialized) {
            return true;
        }
        
        const input = inputs[0];
        if (input && input[0]) {
            const inputData = input[0];
            
            // Add input data to buffer
            for (let i = 0; i < inputData.length; i++) {
                this.buffer[this.bufferIndex] = inputData[i];
                this.bufferIndex++;
                
                // When buffer is full, send chunk for processing
                if (this.bufferIndex >= this.chunkSize) {
                    const chunk = new Float32Array(this.buffer.subarray(0, this.chunkSize));
                    
                    this.port.postMessage({
                        type: 'audioChunk',
                        data: chunk,
                        timestamp: currentTime
                    });
                    
                    // Move remaining data to beginning of buffer
                    const remaining = this.bufferIndex - this.chunkSize;
                    if (remaining > 0) {
                        this.buffer.copyWithin(0, this.chunkSize, this.bufferIndex);
                    }
                    this.bufferIndex = remaining;
                }
            }
        }
        
        return true;
    }
}

registerProcessor('voirs-audio-processor', VoirsAudioProcessor);
"#
    .to_string()
}

#[wasm_bindgen]
/// get streaming capabilities
pub fn get_streaming_capabilities() -> JsValue {
    let capabilities = serde_json::json!({
        "real_time_processing": true,
        "chunked_recognition": true,
        "overlapping_windows": true,
        "adaptive_quality": true,
        "buffer_management": true,
        "latency_monitoring": true,
        "audio_worklets": true,
        "web_audio_api": true,
        "performance_metrics": true,
        "sample_rates": [8000, 16000, 22050, 44100, 48000],
        "chunk_sizes": ["adaptive", "256", "512", "1024", "2048", "4096"],
        "latency_targets": ["ultra_low", "low", "medium", "high"]
    });

    JsValue::from_serde(&capabilities).unwrap_or(JsValue::NULL)
}
