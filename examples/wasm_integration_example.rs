//! WebAssembly Integration Example - Browser-Based VoiRS Speech Synthesis
//!
//! This example demonstrates how to integrate VoiRS with WebAssembly for browser-based
//! text-to-speech applications. It showcases the key patterns and considerations needed
//! for running VoiRS synthesis in web browsers.
//!
//! ## What this example demonstrates:
//! 1. WebAssembly-compatible VoiRS configuration
//! 2. Browser-friendly audio handling and output
//! 3. Async/await patterns optimized for WASM
//! 4. Memory management considerations for browsers
//! 5. Error handling in WASM environment
//! 6. Performance optimization for web deployment
//!
//! ## Key Features for Web Integration:
//! - WASM-compatible synthesis pipeline
//! - Browser audio context integration
//! - Chunk-based processing for responsive UI
//! - Memory-efficient operations
//! - Error handling with user-friendly messages
//! - Performance monitoring and optimization
//!
//! ## Prerequisites:
//! - Rust with wasm-pack installed
//! - wasm-bindgen and web-sys dependencies
//! - Modern web browser with WebAssembly support
//! - HTTP server for serving WASM files
//!
//! ## Building for WebAssembly:
//! ```bash
//! wasm-pack build --target web --out-dir pkg
//! ```
//!
//! ## Expected output:
//! - WebAssembly module ready for browser deployment
//! - JavaScript bindings for VoiRS functionality
//! - Example HTML/JS integration code

use anyhow::{Context, Result};
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use web_sys::{console, AudioBuffer, AudioBufferSourceNode, AudioContext};

use voirs::*;

// WebAssembly-specific error handling
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! console_log {
    ($($t:tt)*) => (println!($($t)*))
}

/// WebAssembly-compatible VoiRS synthesizer
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmVoirsSynthesizer {
    pipeline: VoirsPipeline,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmVoirsSynthesizer {
    /// Create a new WebAssembly VoiRS synthesizer
    #[wasm_bindgen(constructor)]
    pub async fn new() -> Result<WasmVoirsSynthesizer, JsValue> {
        console_log!("🌐 Initializing WebAssembly VoiRS Synthesizer");

        // Create lightweight components optimized for WASM
        let g2p = create_g2p(G2pBackend::RuleBased);
        let acoustic = create_acoustic(AcousticBackend::Vits); // Use lighter model in production
        let vocoder = create_vocoder(VocoderBackend::HifiGan);

        let pipeline = VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .build()
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to build WASM pipeline: {}", e)))?;

        console_log!("✅ WebAssembly synthesizer ready");
        Ok(WasmVoirsSynthesizer { pipeline })
    }

    /// Synthesize text to audio in WebAssembly environment
    #[wasm_bindgen]
    pub async fn synthesize(&self, text: &str) -> Result<js_sys::Float32Array, JsValue> {
        console_log!("🎵 Synthesizing in WebAssembly: '{}'", text);

        let start_time = Instant::now();

        let audio = self
            .pipeline
            .synthesize(text)
            .await
            .map_err(|e| JsValue::from_str(&format!("Synthesis failed: {}", e)))?;

        let synthesis_time = start_time.elapsed();
        console_log!(
            "✅ Synthesis complete ({:.2}s, RTF: {:.2}x)",
            synthesis_time.as_secs_f32(),
            synthesis_time.as_secs_f32() / audio.duration()
        );

        // Convert audio samples to JavaScript Float32Array
        let samples = audio.samples();
        let js_array = js_sys::Float32Array::new_with_length(samples.len() as u32);
        js_array.copy_from(samples);

        Ok(js_array)
    }

    /// Get audio information
    #[wasm_bindgen]
    pub async fn get_audio_info(&self, text: &str) -> Result<JsValue, JsValue> {
        let audio = self
            .pipeline
            .synthesize(text)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to get audio info: {}", e)))?;

        let info = serde_wasm_bindgen::to_value(&serde_json::json!({
            "sample_rate": audio.sample_rate(),
            "duration": audio.duration(),
            "channels": audio.channels(),
            "samples": audio.samples().len()
        }))
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize audio info: {}", e)))?;

        Ok(info)
    }

    /// Create Web Audio API compatible audio buffer
    #[wasm_bindgen]
    pub async fn create_audio_buffer(
        &self,
        text: &str,
        audio_context: &AudioContext,
    ) -> Result<AudioBuffer, JsValue> {
        let audio = self.pipeline.synthesize(text).await.map_err(|e| {
            JsValue::from_str(&format!("Failed to synthesize for audio buffer: {}", e))
        })?;

        // Create Web Audio API buffer
        let buffer = audio_context
            .create_buffer(
                audio.channels() as u32,
                audio.samples().len() as u32,
                audio.sample_rate() as f32,
            )
            .map_err(|e| JsValue::from_str(&format!("Failed to create audio buffer: {:?}", e)))?;

        // Copy audio data to Web Audio buffer
        let channel_data = buffer
            .get_channel_data(0)
            .map_err(|e| JsValue::from_str(&format!("Failed to get channel data: {:?}", e)))?;

        for (i, &sample) in audio.samples().iter().enumerate() {
            channel_data.set_index(i as u32, sample);
        }

        console_log!(
            "✅ Web Audio buffer created: {:.2}s audio",
            audio.duration()
        );
        Ok(buffer)
    }
}

/// Native (non-WASM) example runner
#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() -> Result<()> {
    use tracing::{info, warn};

    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    console_log!("🌐 VoiRS WebAssembly Integration Example");
    console_log!("========================================");
    console_log!();

    console_log!("📝 Note: This example demonstrates WebAssembly integration patterns.");
    console_log!("     When compiled for WASM, it provides browser-compatible bindings.");
    console_log!("     Running natively to show the integration structure.");
    console_log!();

    // Create the synthesizer (demonstrates the WASM-compatible API)
    console_log!("🔧 Creating WASM-compatible synthesizer...");
    let setup_start = Instant::now();

    let g2p = create_g2p(G2pBackend::RuleBased);
    let acoustic = create_acoustic(AcousticBackend::Vits);
    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    let pipeline = VoirsPipelineBuilder::new()
        .with_g2p(g2p)
        .with_acoustic_model(acoustic)
        .with_vocoder(vocoder)
        .build()
        .await
        .context("Failed to build WASM-compatible pipeline")?;

    let setup_time = setup_start.elapsed();
    console_log!(
        "✅ WASM-compatible synthesizer ready in {:.2} seconds",
        setup_time.as_secs_f32()
    );

    // Demonstrate WASM-like synthesis patterns
    let wasm_examples = [
        "Hello from WebAssembly! This speech is generated in the browser.",
        "WebAssembly enables high-performance speech synthesis directly in web applications.",
        "Real-time voice generation is now possible in modern web browsers.",
    ];

    console_log!("\n🎵 WebAssembly Synthesis Demonstration:");
    console_log!("--------------------------------------");

    for (i, text) in wasm_examples.iter().enumerate() {
        console_log!("   Processing WASM example {}...", i + 1);

        let wasm_start = Instant::now();
        let audio = pipeline
            .synthesize(text)
            .await
            .context(format!("Failed to synthesize WASM example {}", i + 1))?;
        let wasm_time = wasm_start.elapsed();

        // Simulate WASM output patterns
        let filename = format!("wasm_example_{:02}.wav", i + 1);
        audio
            .save_wav(&filename)
            .context("Failed to save WASM example audio")?;

        console_log!(
            "   ✅ WASM example {}: {} ({:.2}s, RTF: {:.2}x)",
            i + 1,
            filename,
            wasm_time.as_secs_f32(),
            wasm_time.as_secs_f32() / audio.duration()
        );

        // Show browser-like metrics
        console_log!(
            "      Browser metrics: {}Hz, {:.2}s, {} samples",
            audio.sample_rate(),
            audio.duration(),
            audio.samples().len()
        );
    }

    // WebAssembly integration guidance
    console_log!("\n📋 WebAssembly Integration Guide:");
    console_log!("--------------------------------");
    console_log!("1. Install wasm-pack: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh");
    console_log!("2. Add to Cargo.toml:");
    console_log!("   [dependencies]");
    console_log!("   wasm-bindgen = \"0.2\"");
    console_log!("   web-sys = \"0.3\"");
    console_log!("   js-sys = \"0.3\"");
    console_log!("   serde-wasm-bindgen = \"0.4\"");
    console_log!("3. Build: wasm-pack build --target web");
    console_log!("4. Use in HTML/JavaScript with generated bindings");

    console_log!("\n💡 Browser Integration Pattern:");
    console_log!("------------------------------");
    console_log!("```javascript");
    console_log!("import init, {{ WasmVoirsSynthesizer }} from './pkg/voirs.js';");
    console_log!("await init();");
    console_log!("const synthesizer = await new WasmVoirsSynthesizer();");
    console_log!("const audioData = await synthesizer.synthesize('Hello WebAssembly!');");
    console_log!("```");

    console_log!("\n🎉 WebAssembly Integration Example Complete!");
    console_log!("Generated files: wasm_example_01.wav, wasm_example_02.wav, wasm_example_03.wav");

    Ok(())
}

/// WASM module initialization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    console_log!("🌐 VoiRS WebAssembly module initialized");
}
