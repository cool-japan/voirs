//! # WebAssembly Browser Integration Example
//!
//! This example demonstrates how to use voirs-conversion in a browser environment
//! via WebAssembly, including worker-based processing and real-time audio handling.
//!
//! ## Build Instructions
//! ```bash
//! # Install wasm-pack
//! cargo install wasm-pack
//!
//! # Build for web
//! wasm-pack build --target web --features wasm
//!
//! # Build for Node.js
//! wasm-pack build --target nodejs --features wasm
//! ```
//!
//! ## Features Demonstrated
//! - WebAssembly module initialization
//! - Web Audio API integration
//! - Real-time audio processing in browser
//! - Web Worker for background processing
//! - Memory-efficient streaming
//! - Performance monitoring

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
use voirs_conversion::wasm::{WasmAudioProcessor, WasmConversionConfig, WasmPerformanceMonitor};

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
use web_sys::{AudioContext, AudioContextState};

/// Example main entry point for WASM
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
#[wasm_bindgen(start)]
pub fn main() {
    // Initialize panic hook for better error messages
    console_error_panic_hook::set_once();

    // Initialize logging
    wasm_logger::init(wasm_logger::Config::default());

    log::info!("VoiRS Conversion WASM module initialized");
}

/// Create and configure voice converter for browser
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
#[wasm_bindgen]
pub struct BrowserVoiceConverter {
    processor: WasmAudioProcessor,
    monitor: WasmPerformanceMonitor,
}

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
#[wasm_bindgen]
impl BrowserVoiceConverter {
    /// Create new browser voice converter
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<BrowserVoiceConverter, JsValue> {
        log::info!("Creating browser voice converter");

        let config = WasmConversionConfig {
            sample_rate: 16000,
            chunk_size: 1024,
            enable_streaming: true,
            max_latency_ms: 100,
            memory_limit_mb: 128,
        };

        let processor = WasmAudioProcessor::new(config)
            .map_err(|e| JsValue::from_str(&format!("Failed to create processor: {}", e)))?;

        let monitor = WasmPerformanceMonitor::new();

        Ok(Self { processor, monitor })
    }

    /// Process audio chunk
    #[wasm_bindgen]
    pub fn process_chunk(&mut self, audio_data: Vec<f32>) -> Result<Vec<f32>, JsValue> {
        self.monitor.start_measurement();

        let result = self
            .processor
            .process_chunk(&audio_data)
            .map_err(|e| JsValue::from_str(&format!("Processing error: {}", e)))?;

        self.monitor.end_measurement();

        Ok(result)
    }

    /// Get performance statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> JsValue {
        let stats = self.monitor.get_stats();
        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }

    /// Set conversion target
    #[wasm_bindgen]
    pub fn set_target(&mut self, pitch_factor: f32, gender: f32) -> Result<(), JsValue> {
        self.processor
            .set_conversion_target(pitch_factor, gender)
            .map_err(|e| JsValue::from_str(&format!("Failed to set target: {}", e)))
    }

    /// Reset processor state
    #[wasm_bindgen]
    pub fn reset(&mut self) -> Result<(), JsValue> {
        self.processor
            .reset()
            .map_err(|e| JsValue::from_str(&format!("Failed to reset: {}", e)))
    }
}

/// Example: Create audio context and connect processor
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
#[wasm_bindgen]
pub async fn create_audio_pipeline() -> Result<(), JsValue> {
    log::info!("Creating audio pipeline");

    // Create audio context
    let window = web_sys::window().ok_or("No window found")?;
    let audio_context = AudioContext::new()?;

    // Wait for audio context to be ready
    if audio_context.state() != AudioContextState::Running {
        log::info!("Resuming audio context");
        let resume_promise = audio_context.resume()?;
        wasm_bindgen_futures::JsFuture::from(resume_promise).await?;
    }

    log::info!(
        "Audio context ready, sample rate: {}",
        audio_context.sample_rate()
    );

    Ok(())
}

/// Example: Batch process audio file
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
#[wasm_bindgen]
pub async fn process_audio_file(
    audio_data: Vec<f32>,
    sample_rate: u32,
) -> Result<Vec<f32>, JsValue> {
    log::info!(
        "Processing audio file: {} samples at {}Hz",
        audio_data.len(),
        sample_rate
    );

    let mut converter = BrowserVoiceConverter::new()?;
    converter.set_target(1.2, 0.7)?; // Pitch up, slight female shift

    // Process in chunks
    let chunk_size = 1024;
    let mut result = Vec::with_capacity(audio_data.len());

    for chunk in audio_data.chunks(chunk_size) {
        let processed = converter.process_chunk(chunk.to_vec())?;
        result.extend(processed);
    }

    log::info!("Processing complete: {} output samples", result.len());
    log::info!("Stats: {:?}", converter.get_stats());

    Ok(result)
}

// ============================================================================
// HTML/JavaScript Usage Example
// ============================================================================

/*
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>VoiRS Voice Conversion Demo</title>
</head>
<body>
    <h1>Real-time Voice Conversion</h1>

    <div>
        <label>Pitch Factor: <input type="range" id="pitch" min="0.5" max="2.0" step="0.1" value="1.0"></label>
        <span id="pitch-value">1.0</span>
    </div>

    <div>
        <label>Gender: <input type="range" id="gender" min="-1.0" max="1.0" step="0.1" value="0.0"></label>
        <span id="gender-value">0.0</span>
    </div>

    <button id="start">Start Conversion</button>
    <button id="stop" disabled>Stop</button>

    <div id="stats"></div>

    <script type="module">
        import init, { BrowserVoiceConverter, create_audio_pipeline } from './pkg/voirs_conversion.js';

        let converter;
        let audioContext;
        let processor;
        let isRunning = false;

        // Initialize WASM module
        async function initWasm() {
            await init();
            console.log('WASM module loaded');

            // Create converter
            converter = new BrowserVoiceConverter();
            console.log('Converter created');

            // Setup audio pipeline
            await create_audio_pipeline();
        }

        // Start real-time conversion
        async function startConversion() {
            if (isRunning) return;

            audioContext = new AudioContext();
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            const source = audioContext.createMediaStreamSource(stream);
            const scriptNode = audioContext.createScriptProcessor(1024, 1, 1);

            scriptNode.onaudioprocess = (event) => {
                const inputData = event.inputBuffer.getChannelData(0);
                const inputArray = Array.from(inputData);

                try {
                    const outputArray = converter.process_chunk(inputArray);
                    const outputData = event.outputBuffer.getChannelData(0);

                    for (let i = 0; i < outputArray.length; i++) {
                        outputData[i] = outputArray[i];
                    }

                    // Update stats
                    const stats = converter.get_stats();
                    document.getElementById('stats').textContent =
                        `Latency: ${stats.average_latency_ms.toFixed(2)}ms | ` +
                        `Throughput: ${stats.throughput.toFixed(0)} samples/s`;
                } catch (e) {
                    console.error('Processing error:', e);
                }
            };

            source.connect(scriptNode);
            scriptNode.connect(audioContext.destination);

            isRunning = true;
            document.getElementById('start').disabled = true;
            document.getElementById('stop').disabled = false;
        }

        // Stop conversion
        function stopConversion() {
            if (!isRunning) return;

            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }

            isRunning = false;
            document.getElementById('start').disabled = false;
            document.getElementById('stop').disabled = true;
        }

        // Update conversion parameters
        function updateParams() {
            const pitch = parseFloat(document.getElementById('pitch').value);
            const gender = parseFloat(document.getElementById('gender').value);

            document.getElementById('pitch-value').textContent = pitch.toFixed(1);
            document.getElementById('gender-value').textContent = gender.toFixed(1);

            if (converter) {
                converter.set_target(pitch, gender);
            }
        }

        // Event listeners
        document.getElementById('start').addEventListener('click', startConversion);
        document.getElementById('stop').addEventListener('click', stopConversion);
        document.getElementById('pitch').addEventListener('input', updateParams);
        document.getElementById('gender').addEventListener('input', updateParams);

        // Initialize
        initWasm().catch(console.error);
    </script>
</body>
</html>
*/

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
fn main() {
    println!("This example requires the 'wasm' feature.");
    println!("Build with: wasm-pack build --target web --features wasm");
}
